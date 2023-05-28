from __future__ import annotations

import os
from operator import itemgetter
from typing import Optional, Union

import adapmen
import numpy as np
import torch
from unstable_baselines.common.networks import SequentialNetwork

from adapmen.env.human_in_the_loop_env import HumanInTheLoopEnv
from adapmen.env.expert_guided_env import ExpertGuidedEnv
import gym
#from adapmen.env.safe_env import BenchmarkEnv
from adapmen.env import  MUJOCO_ENVS, ATARI_ENVS, METADRIVE_ENVS 

def get_expert( env_name, device, mean, std, env):
    if env_name in MUJOCO_ENVS:
        return SACExpert(env, env_name, device, mean, std)
    elif env_name in METADRIVE_ENVS:
        return SACExpert(env, "metadrive", device, mean, std)
    elif env_name in ATARI_ENVS:
        return DQNExpert(env, env_name, device, mean, std)
    else:
        raise NotImplementedError


class SACExpert:
    def __init__(self, env, env_name: str, device: str, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        #data_root = "/" + os.path.join( *(adapmen.__file__.split(os.sep)[:-2] + ['data', 'expert_model']))
        '''
        data_root =  os.path.join( *(adapmen.__file__.split(os.sep)[:-2] + ['data', 'expert_model']))
        agent_model_path = os.path.join(data_root, env_name+".pt")
        
        self.device = torch.device(device)
        from unstable_baselines.baselines.sac.configs.metadrive.default import default_args
        #default_args['agent']['gamma'] = default_args['common']['gamma']
        from unstable_baselines.baselines.sac.agent import SACAgent
        self.agent = SACAgent(env.observation_space, env.action_space, **default_args['agent'])
        self.agent.load_state_dict(torch.load(agent_model_path, map_location=self.device))
        self.q_network = self.agent.q1_network.to(self.device)
        self.policy_network = self.agent.policy_network.to(self.device)
        '''

        q_path = os.path.join('data/expert_model', env_name) + '/q_network.pt'
        #q_path = os.path.join('data/expert_model', env_name) + '/q_network_expert.pt'
        policy_path = os.path.join('data/expert_model', env_name) + '/policy_network.pt'
        #policy_path = os.path.join('data/expert_model', env_name) + '/policy_network_expert_45w.pt'

        self.device = torch.device(device)

        self.q_network = torch.load(q_path,map_location='cuda')
        self.policy_network = torch.load(policy_path,map_location='cuda')
        self.q_network.to(self.device)
        self.policy_network.to(self.device)


        if mean is not None and std is not None:
            self.mean = torch.from_numpy(mean).float().to(self.device)
            self.std = torch.from_numpy(std).float().to(self.device)
        else:
            self.std = 1.0
            self.mean = 0.0

    def denormalize(self, state):
        return state * (self.std + 1e-3) + self.mean

    def select_action(self, state: torch.Tensor, deterministic=True):
        with torch.no_grad():
            state = self.denormalize(state)
            action_scaled, log_prob, log_std = \
                itemgetter("action_scaled", "log_prob", 'log_std')(
                    self.policy_network.sample(state, deterministic))

        return {
            "action": action_scaled.cpu().squeeze().numpy(),
            "log_prob": log_prob.cpu().numpy(),
            'log_std': log_std.cpu().numpy()
        }

    def forward(self, state):
        action_scaled, log_prob, log_std = \
                itemgetter("action_scaled", "log_prob", 'log_std')(
                    self.policy_network.sample(state, deterministic=True))

        return {
            "action": action_scaled.squeeze(),
            "log_prob": log_prob.squeeze(),
            'log_std': log_std.squeeze()
        }

    def predict_q_value(self, state, action):
        state = self.denormalize(state)
        q_value = self.q_network(torch.cat([state, action], dim=1))
        return q_value


class DQNExpert:
    def __init__(self, env, env_name: str, device: str, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray]=None):
        data_root = "/" + os.path.join( *(adapmen.__file__.split(os.sep)[:-2] + ['data', 'expert_model', 'atari']))
        agent_model_path = os.path.join(data_root, env_name+".pt")
        
        self.device = device

        from unstable_baselines.baselines.dqn.configs.atari.default import default_args
        default_args['agent']['gamma'] = default_args['common']['gamma']
        from unstable_baselines.baselines.dqn.agent import DQNAgent
        self.agent = DQNAgent(env.observation_space, env.action_space, **default_args['agent'])
        self.agent.load_state_dict(torch.load(agent_model_path, map_location=self.device))
        
        self.q_network = self.agent.q_network.to(self.device)

        if mean is not None and std is not None:
            self.mean = torch.from_numpy(mean).float().to(self.device)
            self.std = torch.from_numpy(std).float().to(self.device)
        else:
            self.std = 1.0
            self.mean = 0.0

    def denormalize(self, state):
        return state * (self.std + 1e-3) + self.mean

    def select_action(self, obs):
        obs = self.denormalize(obs)
        if len(obs.shape) == 4:
            obs = obs / 255.0
        q_values = self.q_network(obs)
        q_values, action_indices = torch.max(q_values, dim=1)
        action = action_indices.detach().cpu().numpy()[0]
        return {"action": action}

    def predict_q_value(self, state, action):
        state = self.denormalize(state)
        if len(state.shape) == 4:
            state = state / 255.0
        #print(action, action.int())
        q_value = self.q_network(state)
        q_value = q_value[0][action.long()]
        return q_value
    
    def forward(self, state):
        state = self.denormalize(state)
        if len(state.shape) == 4:
            state = state / 255.0
        q_values = self.q_network(state)
        q_values, action_indices = torch.max(q_values, dim=1)
        action = action_indices.detach().cpu().numpy()[0]
        return {"action": action, "log_std": None}

class SafeExpert:
    def __init__(self, policy_path, q_path, env, device=torch.device("cpu")):
        # Safety layer.
        self.safety_layer = SafetyLayer(env.observation_space,
                                        env.action_space,
                                        hidden_dim=64,
                                        num_constraints
                                        =env.num_constraints,
                                        lr=0.0001,
                                        slack=env.slack,
                                        device=device
                                        )
        # Agent.
        self.agent = SafePPOAgent(
            env.observation_space,
            env.action_space,
            hidden_dim=env.hidden_dim,
            action_modifier=self.safety_layer.get_safe_action,
            device=device
        )
        # Pre-/post-processing.
        self.obs_normalizer = BaseNormalizer()
        self.reward_normalizer = BaseNormalizer()
        self.load(policy_path, device)
        self.q_network =  SequentialNetwork(env.observation_space.shape[0] + env.action_space.shape[0], 1, [('mlp', 256), ('mlp', 256)])
        self.q_network.to(device)
        self.q_network.load_state_dict(torch.load(q_path, map_location=device))
        self.num_constraints = env.num_constraints

    def select_action(self, obs, c):
        action = self.agent.ac.act(obs, c=c).squeeze()

        return {"action": action}


    def predict_q_value(self, obs, action):
        q_value = self.q_network(torch.cat([obs, action], dim=1))
        return q_value

    def load(self,
             path,
             device
             ):
        """Restores model and experiment given checkpoint path.

        """
        state = torch.load(path, map_location=device)
        # Restore policy.
        self.agent.load_state_dict(state["agent"])
        self.safety_layer.load_state_dict(state["safety_layer"])
        self.obs_normalizer.load_state_dict(state["obs_normalizer"])
        self.reward_normalizer.load_state_dict(state["reward_normalizer"])
    

if __name__ == '__main__':
    from tqdm import trange
    # env_name = 'HalfCheetah-v3'
    # q_path = os.path.join('data/expert_model', env_name) + '/q1_network.pt'
    # policy_path = os.path.join('data/expert_model', env_name) + '/policy_network.pt'
    # q_network = torch.load(q_path)
    # policy_network = torch.load(policy_path)
    from adapmen.env import create_safe_env
    # model_path = "../../data/expert_model/quadrotor-safe/policy_network.pt"
    # q_path = "../../data/expert_model/quadrotor-safe/q1.pt"
    # env = create_safe_env("quadrotor_constraint", 100)
    model_path = "../../data/expert_model/cartpole-safe/policy_network.pt"
    q_path = "../../data/expert_model/cartpole-safe/q1.pt"
    env = create_safe_env("cartpole_constraint", 100)
    agent = SafeExpert(model_path, q_path, env,)

    traj_rets = []
    for traj in trange(100):
        obs, info = env.reset()
        done = False
        c = info["constraint_values"]
        traj_ret = 0
        while not done:
            action = agent.select_action(torch.FloatTensor(obs), c=torch.FloatTensor(c))['action']
            next_obs, reward, done, info = env.step(action)
            traj_ret += reward
            obs = next_obs
            c = info['constraint_values']
            if done:
                break
        traj_rets.append(traj_ret)
    print(np.mean(traj_rets), np.std(traj_rets), min(traj_rets), max(traj_rets))
    import matplotlib.pyplot as plt
    plt.hist(traj_rets)
    plt.savefig("temp.png")


