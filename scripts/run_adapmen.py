import numpy as np
import torch
import os
import time
import click
from adapmen.algo import evaluate
from adapmen.algo.il.btq import BTQ, DiscreteBTQ
from adapmen.buffer import ExpertBuffer, OffPolicyBuffer
from adapmen.utils.config_loader import ConfigLoader
from unstable_baselines.common.util import load_config
from adapmen.env import create_il_env, METADRIVE_ENVS, EXPERT_PERFORMANCE
from adapmen.model import Actor, DiscreteActor
from adapmen.model.expert import get_expert
from adapmen.model.expert import get_expert
from adapmen.utils import set_seed, dict_to_list, init_logging, log_and_write
from operator import itemgetter

import munch
@click.command()
@click.argument("config_path", type=str)
@click.option('--gpu', type=int, default=-1)
@click.option('--seed', type=int, default=0)
@click.option('--info', type=str, default='')
def main(config_path, gpu, seed, info):
    hparams_dict = load_config(config_path)
    cfg = munch.munchify(hparams_dict)
    writer, log_dir, save_dir = init_logging(cfg, hparams_dict, info)

    set_seed(seed)
    device = torch.device("cuda:{}".format(gpu)) if gpu>=0 else torch.device("cpu")
    
    #initialize env
    env = create_il_env(cfg.env.env_name, seed, None, None, False)
    #eval_env = create_il_env(cfg.env.env_name, seed+1, None, None, False)
    
    #initialize buffer
    expert_buffer = ExpertBuffer(cfg.env.expert_data_path, cfg.env.num_traj, device)
    policy_buffer = OffPolicyBuffer(cfg.buffer_size, env.observation_space,
                                env.action_space, device)
    
    if cfg.env.use_normalize_states and cfg.env.num_traj > 0:
        mean, std = expert_buffer.normalize_states()
    else:
        mean, std = None, None

    #initialize agent
    
    if env.discrete:
        actor = DiscreteActor(env.observation_space, env.action_space.n, cfg.sac.actor_network_params)
        actor.to(device)
        imitator = DiscreteBTQ(actor, cfg)
    else:
        actor = Actor(env.observation_space, env.action_space.shape[0], cfg.sac.actor_network_params)
        actor.to(device)
        imitator = BTQ(actor, cfg)
    if cfg.env.env_name == 'metadrive':
        eval_env = env
    else:
        eval_env = create_il_env(cfg.env.env_name, seed+1, mean, std, False)
    sac = imitator

    expert = get_expert(cfg.env.env_name, device, mean, std, env)
    
    if cfg.env.num_traj > 0:
        policy_buffer.insert_expert_buffer(expert_buffer)

    episode_return = 0
    episode_timesteps = 0
    done = True
    success_num = 0
    episode_num = 0
    episode_cost = 0
    expert_intervention = 0
    totsteps_intervention_map = []   
    totsteps_intervention_map.append(0)

    total_timesteps = 0 
    batch_actor_loss = 0.01
    p_value = cfg.btq.p_value
    eval_returns = []
    p_value_list = []

    eval_returns = []
    expert_ret = EXPERT_PERFORMANCE.get(cfg.env.env_name, "1e10")
    for step in range(cfg.btq.start_training_timesteps):
        policy_buffer_gen = policy_buffer.get_batch_generator_inf(cfg.btq.batch_size)
        for _ in range(cfg.btq.updates_per_step):
            infos = imitator.update(next(policy_buffer_gen))
        if total_timesteps % cfg.log_interval == 0:
            log_infos = dict_to_list(infos)
            log_and_write(writer, log_infos, global_step=total_timesteps)

    while total_timesteps < cfg.btq.max_timesteps:

        if cfg.env.env_name not in METADRIVE_ENVS and total_timesteps % cfg.eval_interval == 0:
            average_return, average_length = itemgetter("average_return", "average_length")(evaluate(actor, eval_env, device))
            log_infos = [('timestep', total_timesteps),
                         ('perf/eval_return', average_return),
                         ('perf/eval_length', average_length),
                         ('perf/eval_return_ratio', average_return / expert_ret)]
            log_and_write(writer, log_infos, global_step=total_timesteps)
            intervetion_log = [('perf/intervetion_ret', average_return)]
            log_and_write(writer, intervetion_log, global_step=expert_intervention)

            eval_returns.append(average_return)
            # np.save(save_dir, np.array(eval_returns))

        if done: 
            episode_num += 1
            log_infos = [('perf/train_return', episode_return)]
            log_and_write(writer, log_infos, global_step=episode_num)

            if episode_timesteps > 0:
                current_time = time.time()
            if hasattr(env, "is_safe_env") and env.is_safe_env:
                state, info = env.reset()
                c = info["constraint_values"]
            else:
                state = env.reset()
                info = {'dagger_action': [0,0]}
            episode_cost = 0
            episode_return = 0
            episode_timesteps = 0
        if env.discrete:
            action_samples, action_modes = itemgetter("sample", "mode")(sac.actor(torch.from_numpy(np.array([state], dtype=np.float32)).to(device)))
            action = action_samples.squeeze().detach().cpu().numpy()
        else:
            action_mean, action_log_std = itemgetter("mean", "log_std")(sac.actor(torch.from_numpy(np.array([state], dtype=np.float32)).to(device)))
            action = action_mean.squeeze().detach().cpu().numpy()
        #action = (action + np.random.normal(0, 0.1, size=action.shape)).clip(-1, 1)
        # if hasattr(env, "is_safe_env") and env.is_safe_env:
        #     expert_action = expert.select_action(torch.from_numpy(np.array([state], dtype=np.float32)).to(device), torch.from_numpy(np.array([c], dtype=np.float32)).to(device))['action']
        # else:
        if cfg.env.env_name == "HumanDAggerMetaDrive":
            expert_action = info['dagger_action']
        else:
            expert_action_mean, expert_action_log_std = itemgetter("action", "log_std")(expert.forward(torch.from_numpy(np.array([state], dtype=np.float32)).to(device)))
            if env.discrete:
                expert_action = expert_action_mean
            else:
                expert_action = expert_action_mean.detach().cpu().numpy()

        takeover = False
        if cfg.btq.criterion_type == 'q':
            expert_q_value = expert.predict_q_value(torch.from_numpy(np.array([state], dtype=np.float32)).to(device),
                                                    torch.from_numpy(np.array([expert_action], dtype=np.float32)).to(device))
            policy_q_value = expert.predict_q_value(torch.from_numpy(np.array([state], dtype=np.float32)).to(device),
                                                    torch.from_numpy(np.array([action], dtype=np.float32)).to(device))
            
            q_diff = abs(expert_q_value - policy_q_value)
            # q_diff = expert_q_value - policy_q_value  #
            if q_diff >= p_value:
                takeover = True
        elif cfg.btq.criterion_type == 'pi':
            expert_dist = torch.distributions.Normal(expert_action_mean, expert_action_log_std.exp())
            agent_dist = torch.distributions.Normal(action_mean, action_log_std.exp())
            pi_diff = torch.distributions.kl.kl_divergence(agent_dist, expert_dist).mean().item()
            #pi_diff = np.sqrt(pi_diff / 2)
            if pi_diff >= p_value:
                takeover=True
        else:
            raise NotImplementedError

        if takeover:
            expert_intervention += 1
            next_state, reward, done, info= env.step(expert_action)
            policy_buffer.insert(state, expert_action, next_state, np.array([0]), np.array([1.0]))    
        else:
            next_state, reward, done, info= env.step(action)
            if cfg.btq.insert_all:
                policy_buffer.insert(state, expert_action, next_state, np.array([0]), np.array([1.0]))    
            # policy_buffer.insert(state, action, next_state, np.array([0]), np.array([1.0]))
        # elif q_d""iff < 3 * cfg.btq.p_value:
        #     expert_intervention += 1
        #     next_state, reward, done, _ = env.step(action)
        #     policy_buffer.insert(state, expert_action, next_state, np.array([0]), np.array([1.0]))

        if done:
            if info.get('arrive_dest', 0):
                success_num += 1

        if total_timesteps % cfg.log_interval == 0:
            if episode_num==0:
                suc_rate = 0
            else:
                suc_rate = success_num/episode_num
            log_infos = [
                        ('train/total_success_rate', suc_rate)
                        ]
            log_and_write(writer, log_infos, global_step=total_timesteps)

            if cfg.btq.criterion_type == 'q':
                log_infos = [('train/q_diff', q_diff.item()), ('train/p_value', p_value),
                            ('train/expert_intervention', expert_intervention)
                            ]
            elif cfg.btq.criterion_type == 'pi':
                log_infos = [('train/pi_diff', pi_diff), ('train/p_value', p_value),
                            ('train/expert_intervention', expert_intervention)
                            ]
            else:
                raise NotImplementedError
            log_and_write(writer, log_infos, global_step=total_timesteps)
        totsteps_intervention_map.append( expert_intervention )
        episode_return += reward
        episode_cost += info.get('cost', 0)
        episode_timesteps += 1
        total_timesteps += 1

        state = next_state


        if episode_timesteps > 2000:   
            done = True
        
        # if total_timesteps == cfg.btq.samplesteps:
        #     old_expert_intervention = expert_intervention

        if total_timesteps > cfg.btq.samplesteps and total_timesteps % cfg.log_interval == 0:   
            batch_actor_loss = 0
            for _ in range(cfg.btq.updates_per_step):
                policy_buffer_gen = policy_buffer.get_batch_generator_inf(cfg.btq.batch_size)
                infos = imitator.update(next(policy_buffer_gen))
                log_infos = dict_to_list(infos)  
                batch_actor_loss += log_infos[0][1]
            batch_actor_loss = batch_actor_loss / cfg.btq.updates_per_step

            if total_timesteps % cfg.log_interval == 0:
                # log_and_write(writer, log_infos, global_step=total_timesteps)
                torch.save(actor.actor_model.state_dict(), os.path.join(save_dir, '{}-{}'.format(total_timesteps, expert_intervention)))
            
            if total_timesteps % cfg.btq.update_p_freq == 0:   # update_p_value()

                expert_intervention_prob = (expert_intervention - totsteps_intervention_map[total_timesteps-1000]) / 1000

                if cfg.btq.criterion_type == 'q':
                    p_value = batch_actor_loss * expert_intervention_prob * cfg.horizon
                elif cfg.btq.criterion_type == 'pi':
                    state_batch = next(policy_buffer.get_batch_generator_epoch(policy_buffer.size))['state']
                    actor_action_mean, actor_log_std = itemgetter('mean', 'log_std')(actor(state_batch))
                    expert_action_mean, expert_action_log_std = itemgetter("action", "log_std")(expert.forward(state_batch))
                    expert_dist = torch.distributions.Normal(expert_action_mean, expert_action_log_std.exp())
                    agent_dist = torch.distributions.Normal(actor_action_mean, actor_log_std.exp())
                    pi_diff = torch.distributions.kl.kl_divergence(agent_dist, expert_dist).mean().item()
                    #pi_diff = np.sqrt(pi_diff/2)
                    p_value = pi_diff * expert_intervention_prob * cfg.horizon
                p_value_list.append(p_value)
    

if __name__ == "__main__":
    main()