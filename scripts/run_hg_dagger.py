import numpy as np
import torch
import time
import click

from adapmen.algo import DAgger, evaluate
from adapmen.buffer import ExpertBuffer, OffPolicyBuffer
from unstable_baselines.common.util import load_config
import munch
from adapmen.env import create_il_env
from adapmen.model.actor import EnsembleActor
from adapmen.model.expert import get_expert
from adapmen.utils import set_seed, dict_to_list, init_logging, log_and_write
from operator import itemgetter
import os

tot_bc_steps = 0


def train_agent(actor, buffer, num_epochs, batch_size, optimizers, writer):
    global tot_bc_steps
    actor.train()
    loss_fn = torch.nn.MSELoss()
    policy_buffer_gen = buffer.get_batch_generator_inf(batch_size)
    for epoch in range(num_epochs):
        losses = [[] for _ in optimizers]
        batch = next(policy_buffer_gen)
        states, expert_action = itemgetter('state', 'action')(batch)
        actor_actions = itemgetter('actions')(actor(states))
        for i, (actor_action, act, optim) in enumerate(zip(actor_actions, actor.actor_models, optimizers)):
            loss = loss_fn(actor_action, expert_action)
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses[i].append(loss.item())
            if i == 0:
                tot_bc_steps += 1
                log_infos = [('bc/loss', loss.item()),]
                log_and_write(writer, log_infos, global_step=tot_bc_steps)
    losses = [np.mean(l) for l in losses]
    return losses

@click.command()
@click.argument("config_path", type=str)
@click.option("--seed", type=int, default=12345)
@click.option('--gpu', type=int, default=-1)
def main(config_path, seed, gpu):
    hparams_dict = load_config(config_path)
    cfg = munch.munchify(hparams_dict)
    writer, log_dir, save_dir = init_logging(cfg, hparams_dict)

    set_seed(seed)
    device = torch.device("cuda:{}".format(gpu)) if gpu>=0 else torch.device("cpu")

    expert_buffer = ExpertBuffer(cfg.env.expert_data_path, cfg.env.num_traj, device)
    if cfg.env.use_normalize_states and cfg.env.num_traj > 0:
        mean, std = expert_buffer.normalize_states()
    else:
        mean, std = None, None

    env = create_il_env(cfg.env.env_name, seed, mean, std, False)
    env.reset()    
    #eval_env = create_il_env(cfg.env.eval_env_name, seed+1, mean, std, False)
    #print(type(env), type(env.engine), type(eval_env), type(eval_env.engine))
    #eval_env.reset()
    #exit(0)
    actor = EnsembleActor(env.observation_space.shape[0], env.action_space.shape[0], cfg.actor.actor_hidden_dims, num_actors=cfg.actor.num_actors)
    actor.to(device)
    optimizers = [torch.optim.SGD(actor_model.parameters(), lr=cfg.train.learning_rate ,weight_decay=1e-5) for actor_model in actor.actor_models]
    expert = get_expert(cfg.env.env_name, device, mean, std, env)

    policy_buffer = OffPolicyBuffer(cfg.buffer_size, env.observation_space.shape[0],
                                env.action_space.shape[0], device)
    
    if cfg.env.num_traj > 0:
        policy_buffer.insert_expert_buffer(expert_buffer)
        #supervised training step
        train_agent(actor, policy_buffer, num_epochs=cfg.train.num_epochs, batch_size=cfg.train.batch_size, optimizers=optimizers)

    episode_return = 0
    num_sampling_epochs = 0
    done = True
    takeover_count = 0

    total_timesteps = 0

    expert_intervention = 0
    done_num = 0
    success_num = 0
    actor = EnsembleActor(env.observation_space.shape[0], env.action_space.shape[0], cfg.actor.actor_hidden_dims, num_actors=cfg.actor.num_actors)
    actor.to(device)
    optimizers = [torch.optim.SGD(actor_model.parameters(), lr=cfg.train.learning_rate ,weight_decay=1e-5) for actor_model in actor.actor_models]
    #dagger step
    while total_timesteps < cfg.train.max_timesteps:
        episode_length = 0
        episode_return = 0
        
        episode_cost = 0
        
        state = env.reset()
        # for user friendly :)
        #env.env_method("stop")
        print("Finish training iteration:{} current total timestep:{} takeover count: {}".format(num_sampling_epochs - 1, total_timesteps, takeover_count))

        while True:
            mean_action = actor.act(torch.from_numpy(np.array([state], dtype=np.float32)).to(device))['mean']
            action = mean_action.squeeze()
            #print(mean_action)
            next_state, r, done, info = env.step(action)
            action = info["raw_action"]
            takeover = info["takeover"]
            episode_return += r
            episode_cost += info["native_cost"]
            total_timesteps += 1
            if takeover:
                # for hg dagger aggregate data only when takeover occurs
                policy_buffer.insert(state, list(action), next_state, [r], [done])
                takeover_count += 1

            state = next_state
            episode_length += 1
            

            if total_timesteps % cfg.train.train_agent_interval == 0 and total_timesteps > cfg.train.samplesteps:
                print("training agent",policy_buffer.size)
                #train old agent
                train_agent(actor, policy_buffer, cfg.train.num_epochs, batch_size=cfg.train.batch_size, optimizers=optimizers, writer=writer)


            if total_timesteps % cfg.snapshot_interval == 0 and total_timesteps > cfg.train.samplesteps:
                torch.save(actor, os.path.join(save_dir, '{}-{}.pt'.format(total_timesteps, takeover_count)))

            if done:
                if info['arrive_dest']:
                    success_num += 1
                done_num += 1

            if total_timesteps % cfg.log_interval == 0:
                print("totaltimestep: ",total_timesteps)
                if done_num == 0:
                    suc_rate = 0
                else:
                    suc_rate = success_num / done_num
                log_infos = [
                            ('train/total_success_rate', suc_rate)
                            ]
                log_and_write(writer, log_infos, global_step=total_timesteps)  


            if done:
                log_infos = [('perf/train_return', episode_return),
                            ('perf/train_length', episode_length)
                            ]
                log_and_write(writer, log_infos, global_step=total_timesteps)  
                log_infos = [('perf/intervention_train_return', episode_return),
                            ('perf/intervention_train_length', episode_length)
                            ]
                log_and_write(writer, log_infos, global_step=takeover_count)    
                break
            

        num_sampling_epochs += 1


if __name__ == '__main__':
    main()