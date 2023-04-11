import imp
import torch
import metadrive

from adapmen.algo import BC, evaluate, DiscreteBC
from adapmen.buffer import ExpertBuffer
import munch

from adapmen.utils.config_loader import ConfigLoader#, load_config
from unstable_baselines.common.util import load_config
from adapmen.env import ATARI_ENVS, create_il_env, NormalizeBoxActionWrapper
from adapmen.model import Actor, DiscreteActor
from adapmen.utils import set_seed, dict_to_list, init_logging, log_and_write
import click
from operator import itemgetter
from adapmen.env import EXPERT_PERFORMANCE
import os
from tqdm import trange

@click.command()
@click.argument('config_path', type=str)
@click.option('--gpu', type=int, default=-1)
@click.option('--seed', type=int, default=0)
@click.option('--info', type=str, default="")
def main(config_path, gpu, seed, info):
    hparams_dict = load_config(config_path)
    cfg = munch.munchify(hparams_dict)
    writer, log_dir, save_dir = init_logging(cfg, hparams_dict, info_str=info)

    set_seed(seed)
    device = torch.device("cuda:{}".format(gpu)) if gpu>=0 else torch.device("cpu")

    expert_buffer = ExpertBuffer(cfg.env.expert_data_path, cfg.env.num_traj, device, cfg.env.subsample_rate)
    mean, std = None, None
    
    env = create_il_env(cfg.env.env_name, seed, mean, std, False)


    expert_buffer_gen = expert_buffer.get_batch_generator_inf(cfg.bc.batch_size)
    if env.discrete:
        actor = DiscreteActor(env.observation_space, env.action_space.n, cfg.sac.actor_network_params)
        actor.to(device)
        imitator = DiscreteBC(actor, cfg)
    else:
        actor = Actor(env.observation_space, env.action_space.shape[0], cfg.sac.actor_network_params)
        actor.to(device)
        imitator = BC(actor, cfg)
        

    total_timesteps = 0
    eval_returns = []
    expert_ret = EXPERT_PERFORMANCE.get(cfg.env.env_name, 1e10)
    for total_timesteps in trange(cfg.bc.max_timesteps):

        if total_timesteps % cfg.eval_interval == 0:
            average_return, average_c, average_length = itemgetter("average_return", "average_c", "average_length")(evaluate(actor, env, device))
            log_infos = [('timestep', total_timesteps),
                         ('perf/eval_return', average_return),
                         ('perf/eval_length', average_length),
                         ('perf/eval_return_ratio', average_length / expert_ret)]
            log_and_write(writer, log_infos, global_step=total_timesteps)

        infos = imitator.update(next(expert_buffer_gen))

        if total_timesteps % cfg.log_interval == 0:
            log_infos = dict_to_list(infos)
            log_and_write(writer, log_infos, global_step=total_timesteps)
            
        
        if total_timesteps % cfg.snapshot_interval == 0:
            torch.save(actor.actor_model.state_dict(), os.path.join(save_dir, '{}'.format(total_timesteps)))


if __name__ == "__main__":
    main()