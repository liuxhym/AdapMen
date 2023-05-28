from __future__ import annotations

from operator import itemgetter

import numpy as np
import torch
import random
from adapmen.buffer.base_buffer import BaseBuffer


class ExpertBuffer(BaseBuffer):
    def __init__(self, dataset_path: str, num_traj: int, device=torch.device('cpu'), subsample_rate=1.0):
        super(ExpertBuffer, self).__init__()
        self.device = device

        self.states = None
        self.actions = None
        self.next_states = None
        self.dones = None

        self.load(dataset_path, num_traj, subsample_rate=subsample_rate)

    def to(self, device):
        self.device = device

    def load(self, dataset_path: str, num_traj: int, subsample_rate=1.0):
        if num_traj == 0:
            return

        data = np.load(dataset_path, allow_pickle=True)
        states, actions, next_states, dones = itemgetter('states', 'actions', 'next_states', 'dones')(data)
        split_indices = np.where(dones)[0] + 1
        if split_indices[-1] == len(dones):
            split_indices = split_indices[:-1]

        state_trajs, action_trajs, next_state_trajs, done_trajs = \
            np.split(states, split_indices), np.split(actions, split_indices), \
            np.split(next_states, split_indices), np.split(dones, split_indices)

        traj_indices = np.random.choice(np.arange(len(state_trajs)), num_traj)
        selected_state_trajs, selected_action_trajs, selected_next_state_trajs, selected_done_trajs = [], [], [], []
        for traj_index in traj_indices:
            selected_state_trajs.append(state_trajs[traj_index])
            selected_action_trajs.append(action_trajs[traj_index])
            selected_next_state_trajs.append(next_state_trajs[traj_index])
            selected_done_trajs.append(done_trajs[traj_index])

        states = np.concatenate(selected_state_trajs, 0)
        actions = np.concatenate(selected_action_trajs, 0)
        next_states = np.concatenate(selected_next_state_trajs, 0)
        dones = np.concatenate(selected_done_trajs, 0)
        #sub sampling
        tot_indices = range(len(states))
        subsampled_indices= random.sample(tot_indices, int(subsample_rate * len(states)))


        self.states = states.copy()[subsampled_indices]
        self.actions = actions.copy()[subsampled_indices]
        self.next_states = next_states.copy()[subsampled_indices]
        self.dones = dones.copy()[subsampled_indices]
        self.size = self.states.shape[0]
        shape = states[0].shape
        for state in self.states:
            assert state.shape == shape

    def normalize_states(self):
        mean = np.mean(self.states, 0)
        std = np.std(self.states, 0) + 1e-3
        self.states = (self.states - mean) / std
        self.next_states = (self.next_states - mean) / std
        return mean, std

    def add_absorbing_states(self, env):
        states = np.pad(self.states, ((0, 0), (0, 1)), mode='constant')
        next_states = np.pad(self.next_states, ((0, 0), (0, 1)), mode='constant')

        states = list(states)
        actions = list(self.actions)
        next_states = list(next_states)
        dones = list(self.dones)

        i = 0
        cur_len = 0

        while i < self.size:
            cur_len += 1
            if dones[i] and cur_len < env._max_episode_steps:
                cur_len = 0
                absorbing_state = env.get_absorbing_state()
                states.insert(i + 1, absorbing_state.copy())
                next_states[i] = absorbing_state.copy()
                next_states.insert(i + 1, absorbing_state.copy())
                actions.insert(i + 1, np.zeros((env.action_space.shape[0], )))
                dones[i] = 0.0
                dones.insert(i + 1, 1.0)
            i += 1

        self.states, self.actions, self.next_states, self.dones = \
            np.stack(states), np.stack(actions), np.stack(next_states), np.stack(dones)
        self.size = self.states.shape[0]

    def generate_batch_data(self, indices: np.ndarray):
        states = torch.from_numpy(self.states[indices]).float().to(self.device)
        actions = torch.from_numpy(self.actions[indices]).float().to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).float().to(self.device)
        if len(states.shape) == 4:
            states /= 255.0
            next_states /= 255.0

        return {'state': states, 'action': actions, 'next_state': next_states}
