import numpy as np
import torch
import time
import click

from adapmen.algo import DAgger, evaluate, DiscreteDAgger
from adapmen.buffer import ExpertBuffer, OffPolicyBuffer
from unstable_baselines.common.util import load_config
import munch
from adapmen.env import create_il_env, EXPERT_PERFORMANCE
from adapmen.model import Actor, DiscreteActor
from adapmen.model.expert import get_expert
from adapmen.utils import set_seed, dict_to_list, init_logging, log_and_write
from operator import itemgetter
import os
from tqdm import trange
@click.command()
@click.argument("config_path", type=str)
@click.option("--seed", type=int, default=0)
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
    if env.discrete:
        actor = DiscreteActor(env.observation_space, env.action_space.n, cfg.sac.actor_network_params)
        actor.to(device)
        imitator = DiscreteDAgger(actor, cfg)
    else:
        actor = Actor(env.observation_space, env.action_space.shape[0], cfg.sac.actor_network_params)
        actor.to(device)
        imitator = DAgger(actor, cfg)
    env.reset()
    if cfg.env.env_name == 'metadrive':
        eval_env = env
    else:
        eval_env = create_il_env(cfg.env.env_name, seed+1, mean, std, False)
    
    expert = get_expert(cfg.env.env_name, device, mean, std, env)

    policy_buffer = OffPolicyBuffer(cfg.buffer_size, env.observation_space,
                                env.action_space, device)
    
    if cfg.env.num_traj > 0:
        policy_buffer.insert_expert_buffer(expert_buffer)


    sac = imitator

    episode_return = 0
    episode_timesteps = 0
    done = True
    episode_num = 0
    success_num = 0

    total_timesteps = 0

    expert_intervention = 0

    eval_returns = []

    expert_ret = EXPERT_PERFORMANCE.get(cfg.env.env_name, "1e10")
    for step in range(cfg.dagger.bc_step):
        policy_buffer_gen = policy_buffer.get_batch_generator_inf(cfg.dagger.batch_size)
        for _ in range(cfg.dagger.updates_per_step):
            infos = imitator.update(next(policy_buffer_gen))
        if total_timesteps % cfg.log_interval == 0:
            log_infos = dict_to_list(infos)
            log_and_write(writer, log_infos, global_step=total_timesteps)


    
    for  total_timesteps in trange(cfg.dagger.max_timesteps):

        if "metadrive" not in cfg.env.env_name.lower() and total_timesteps % cfg.eval_interval == 0:
            average_return, average_length = itemgetter("average_return", "average_length")(evaluate(actor, eval_env, device))
            log_infos = [('timestep', total_timesteps),
                         ('perf/eval_return', average_return),
                         ('perf/eval_length', average_length),
                         ('perf/eval_return_ratio', average_return / expert_ret)]
            log_and_write(writer, log_infos, global_step=total_timesteps)
            intervetion_log = [('perf/intervetion_ret', average_return)]
            log_and_write(writer, intervetion_log, global_step=total_timesteps)

            eval_returns.append(average_return)
            np.save(save_dir, np.array(eval_returns))

        if done:
            if total_timesteps !=0:
                if info.get('arrive_dest',0):
                    success_num += 1
            episode_num += 1
            log_infos = [('perf/train_return', episode_return)]
            log_and_write(writer, log_infos, global_step=episode_num)
            info = {"dagger_action": [0,0]}
            state = env.reset()
            episode_cost = 0
            episode_return = 0
            episode_timesteps = 0
        if env.discrete:
            action = sac.actor(torch.from_numpy(np.array([state], dtype=np.float32)).to(device))['sample'].squeeze().detach().cpu().numpy()
        else:
            mean_action = sac.actor(torch.from_numpy(np.array([state], dtype=np.float32)).to(device))['mean']
            action = mean_action.squeeze().detach().cpu().numpy()
        # action = (action + np.random.normal(0, 0.1, size=action.shape)).clip(-1, 1)
        if cfg.env.env_name == "HumanDAggerMetaDrive":
            expert_action = info['dagger_action']
        else:
            expert_action = expert.select_action(torch.from_numpy(np.array([state], dtype=np.float32)).to(device))['action']
        next_state, reward, done, info = env.step(action)
        policy_buffer.insert(state, expert_action, next_state, np.array([0]), np.array([1.0]))

        episode_return += reward
        episode_cost += info.get('cost', 0)
        episode_timesteps += 1
        expert_intervention += 1

        state = next_state
        if episode_timesteps > 2000:   
            done = True

        if total_timesteps > cfg.dagger.samplesteps:
            if total_timesteps % cfg.dagger.update_interval == 0:
                for _ in range(cfg.dagger.updates_per_step):
                    policy_buffer_gen = policy_buffer.get_batch_generator_inf(cfg.dagger.batch_size)
                    infos = imitator.update(next(policy_buffer_gen))
            if total_timesteps % cfg.log_interval == 0:
                if episode_num==0:
                    suc_rate = 0
                else:
                    suc_rate = success_num/episode_num
                log_infos = [
                            ('train/total_success_rate', suc_rate)
                            ] 
                log_and_write(writer, log_infos, global_step=total_timesteps)
                log_infos = dict_to_list(infos)
                log_and_write(writer, log_infos, global_step=total_timesteps)
                torch.save(actor.actor_model.state_dict(), os.path.join(save_dir, '{}-{}'.format(total_timesteps, expert_intervention)))

if __name__ == '__main__':
    main()