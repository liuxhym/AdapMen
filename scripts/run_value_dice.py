import time
import os

import torch
import numpy as np

from adapmen.utils.test_metadrive import test_metadrive

from adapmen.algo import ContinuousValueDice, DiscreteValueDice, evaluate
from adapmen.buffer import ExpertBuffer, OffPolicyBuffer, DiscreteOffPolicyBuffer
from unstable_baselines.common.util import load_config
import munch
from adapmen.model import Actor, Estimator, DiscreteActor
from adapmen.utils import set_seed, dict_to_list, init_logging, log_and_write
from adapmen.env import create_il_env, EXPERT_PERFORMANCE, METADRIVE_ENVS
# from adapmen.env.human_in_the_loop_env import HumanInTheLoopEnv
from adapmen.env.expert_guided_env import ExpertGuidedEnv
from adapmen.env.absorbing_wrapper import AbsorbingWrapper
import click
from operator import itemgetter
import adapmen

from adapmen.utils.test_metadrive import test_metadrive

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
    # expert_data_path = "/" + os.path.join( *(adapmen.__file__.split(os.sep)[:-2] + ['data']))
    # expert_data_path = os.path.join(expert_data_path, cfg.env.env_name+".npz")
    expert_data_path = cfg.env.expert_data_path

    expert_buffer = ExpertBuffer(expert_data_path, cfg.env.num_traj, device)
    if cfg.env.use_normalize_states:
        mean, std = expert_buffer.normalize_states()
    else:
        mean, std = None, None
    env = create_il_env(cfg.env.env_name, seed, mean, std, False)
    #env = AbsorbingWrapper(env)
    eval_env = create_il_env(cfg.env.env_name, seed, mean, std, False)
    #eval_env = AbsorbingWrapper(env)
    #expert_buffer.add_absorbing_states(env)
    expert_buffer_gen = expert_buffer.get_batch_generator_inf(cfg.value_dice.batch_size)
    if env.discrete:
        policy_buffer = DiscreteOffPolicyBuffer(cfg.buffer_size, env.observation_space,
                                            env.action_space, device)
    else:
        policy_buffer = OffPolicyBuffer(cfg.buffer_size, env.observation_space,
                                    env.action_space, device)
    nu_net = Estimator(env.observation_space, env.action_space, cfg.value_dice.nu_network_params)
    nu_net.to(device)
    if env.discrete:
        actor = DiscreteActor(env.observation_space, env.action_space.n, cfg.sac.actor_network_params)
        actor.to(device)
        imitator = DiscreteValueDice(actor, nu_net, cfg)
    else:
        actor = Actor(env.observation_space, env.action_space.shape[0], cfg.sac.actor_network_params)
        actor.to(device)
        imitator = ContinuousValueDice(actor, nu_net, cfg)
    

    episode_return = 0
    episode_timesteps = 0
    done = True
    episode_num = 0
    episode_cost = 0

    total_timesteps = 0
    previous_time = time.time()

    eval_returns = []
    expert_ret = EXPERT_PERFORMANCE.get(cfg.env.env_name, "1e10")
    while total_timesteps < cfg.value_dice.max_timesteps:
        
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
            episode_num += 1
            log_infos = [('perf/train_return', episode_return)]
            log_and_write(writer, log_infos, global_step=total_timesteps)

            if episode_timesteps > 0:
                current_time = time.time()

            state = env.reset()
            episode_cost = 0
            episode_return = 0
            episode_timesteps = 0
            previous_time = time.time()

        if total_timesteps < cfg.value_dice.num_random_actions:
            action = env.action_space.sample()
            action = np.array(action)
        else:
            if env.discrete:
                action_samples, action_probs, action_modes = itemgetter("sample", "prob", "mode")(imitator.actor(torch.from_numpy(np.array([state], dtype=np.float32)).to(device)))

                action = action_samples.squeeze().detach().cpu().numpy()
            else:
                mean_action = imitator.actor(torch.from_numpy(np.array([state], dtype=np.float32)).to(device))['mean']
                action = mean_action.squeeze().detach().cpu().numpy()
                action = (action + np.random.normal(0, 0.1, size=action.shape)).clip(-1, 1)

        next_state, reward, done, _ = env.step(action)
        # if episode_timesteps >= 1000:
        #     done = True

        # # done caused by episode truncation.
        # truncated_done = done and episode_timesteps + 1 == env._max_episode_steps  # pylint: disable=protected-access

        # if done and not truncated_done:
        #     next_state = env.get_absorbing_state()
        if env.discrete:
            #policy_buffer.insert(state, torch.zeros(action.shape), action, next_state, np.array([0]), np.array([1.0]))
            policy_buffer.insert(state, np.ones((env.action_space.n,))/ env.action_space.n, next_state, np.array([0]), np.array([1.0]))
        else:
            policy_buffer.insert(state, action, next_state, np.array([0]), np.array([1.0]))

        episode_return += reward
        episode_timesteps += 1
        total_timesteps += 1

        state = next_state

        if total_timesteps >= cfg.value_dice.start_training_timesteps and total_timesteps % cfg.value_dice.updates_interval==0:
            for _ in range(cfg.value_dice.updates_per_step):
                policy_buffer_gen = policy_buffer.get_batch_generator_inf(cfg.value_dice.batch_size)
                infos = imitator.update(next(expert_buffer_gen), next(policy_buffer_gen))

        if total_timesteps >= cfg.value_dice.start_training_timesteps and total_timesteps % cfg.log_interval == 0:
            log_infos = dict_to_list(infos)
            
            log_and_write(writer, log_infos, global_step=total_timesteps)
            torch.save(actor.actor_model.state_dict(), os.path.join(save_dir, '{}-{}'.format(total_timesteps, total_timesteps)))
    
    ## test
    if cfg.env.env_name in METADRIVE_ENVS:
        env.close()
        test_dir = save_dir[:-4]+ 'test'
        test_metadrive(cfg, save_dir, test_dir, device)
                
if __name__ == '__main__':
    main()