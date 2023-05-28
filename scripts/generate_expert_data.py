import os
import gym
import torch
from adapmen.model.expert import SACExpert, DQNExpert, SafeExpert
import numpy as np
from tqdm import tqdm
from time import sleep
import cv2
from adapmen.env import create_il_env, MUJOCO_ENVS, ATARI_ENVS, METADRIVE_ENVS, DISCRETE_ENVS


# env_list = ATARI_ENVS
env_list = ['Pong-v4']
#env_list = ['metadrive']
num_trajs = 100
save_dir = '/data'
device = torch.device('cpu')
seed=np.random.randint(100)

def update_dataset(expert_data, state, action, next_state, done, c=None):
    expert_data['states'].append(state.copy())
    expert_data['actions'].append(action.copy())
    expert_data['next_states'].append(next_state.copy())
    expert_data['dones'].append(done)
    if c is not None:
        expert_data['constraints'].append(c)


def generate_expert_data(env, env_name):
    if env_name in METADRIVE_ENVS:
        return generate_metadrive_expert_data(env, env_name)
    elif env_name in ATARI_ENVS:
        return generate_atari_expert_data(env, env_name)
    else:
        raise NotImplementedError

def generate_atari_expert_data(env, env_name):
    expert_data = {'states': [], 'actions': [], 'next_states': [], 'dones': [], 'constraints': []}
    env = create_il_env(env_name,mean=None, std=None, seed=seed, add_absorbing_state=False)
    expert = DQNExpert(env, env_name, device)
    rets = []
    for episode in tqdm(range(num_trajs)):
        done = False
        state = env.reset()
        ret = 0
        #env.render()
        while not done:
            action = expert.select_action(torch.as_tensor([state], dtype=torch.float32, device=device))['action']
            next_state, reward, done, _ = env.step(action)
            ret += reward
            update_dataset(expert_data, state, action, next_state, done)
            state = next_state
            #env.render()
        rets.append(ret)
    print("{}: min:{:.02f} max:{:.02f} {:.02f}({:.02f})".format(env_name, np.min(rets),  np.max(rets), np.mean(rets), np.std(rets)))
    for k, v in expert_data.items():
        expert_data[k] = np.array(v)

    return expert_data


def generate_metadrive_expert_data(env, senv_name):
    env = create_il_env(env_name, seed=seed, mean=None, std=None, )
    expert_data = {'states': [], 'actions': [], 'next_states': [], 'dones': []}
    expert_cost_list = []
    for episode in tqdm(range(num_trajs)):
        done = False
        expert_reward = 0
        expert_cost = 0
        state = env.reset()
        env.vehicle.expert_takeover = True
        while not done:
            next_state, reward, done, info = env.step([0, 0])
            expert_cost += info['cost']
            expert_reward += reward
            update_dataset(expert_data, state, np.array(info['raw_action']), next_state, done)
            state = next_state
        expert_cost_list.append(expert_reward)
        expert_cost_list.append(expert_cost)
    for k, v in expert_data.items():
        expert_data[k] = np.array(v)
    return expert_data


if __name__ == "__main__":
    for env_name in env_list:
        env = create_il_env(env_name, 0, None, None)
        expert_data = generate_expert_data(env, env_name)
        save_path = os.path.join(save_dir, env_name)
        np.savez(save_path, **expert_data)
    
