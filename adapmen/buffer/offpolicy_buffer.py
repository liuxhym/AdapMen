from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import gym
from adapmen.buffer.base_buffer import BaseBuffer

if TYPE_CHECKING:
    from adapmen.buffer.expert_buffer import ExpertBuffer


class OffPolicyBuffer(BaseBuffer):
    def __init__(self, buffer_size, obs_space, action_space, device=torch.device('cpu')):
        super(OffPolicyBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.device = device

        self.obs_space =  obs_space
        self.action_space = action_space
        self.obs_shape = obs_space.shape
        self.obs_dtype = obs_space.dtype
        if isinstance(action_space, gym.spaces.discrete.Discrete):
            self.action_dim = 1
            #action_dim = action_space.n
            self.discrete_action = True
        elif isinstance(action_space, gym.spaces.box.Box):
            self.action_dim = action_space.shape[0]
            self.discrete_action = False
        self.states = np.zeros((buffer_size,) + self.obs_shape, dtype=self.obs_dtype)
        self.next_states = np.zeros((buffer_size,) + self.obs_shape, dtype=self.obs_dtype)
        if self.discrete_action:
            self.actions = np.zeros((buffer_size, )).astype(np.int8)
        else:
            self.actions = np.zeros((buffer_size,self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1))
        self.masks = np.ones((buffer_size, 1))
        self.constraints = np.zeros((buffer_size, 1))

        self.index = 0
        self.size = 0

    def to(self, device):
        self.device = device

    def insert(self, state: np.ndarray, action: np.ndarray,  next_state: np.ndarray, reward: np.ndarray,
               mask: np.ndarray, constraint: np.ndarray=None):
        self.states[self.index, ] = state.copy()
        self.actions[self.index, ] = action.copy()
        self.next_states[self.index, ] = next_state.copy()
        self.rewards[self.index, ] = reward.copy()
        self.masks[self.index, ] = mask.copy()
        if constraint is not None:
            self.constraints[self.index, ] = constraint.copy()

        self.index = (self.index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def insert_expert_buffer(self, buffer: ExpertBuffer):
        assert self.index == 0 and self.size == 0 and self.buffer_size > buffer.size
        for i in range(buffer.size):
            self.insert(buffer.states[i], buffer.actions[i], buffer.next_states[i], np.array([0]), np.array([1.0]))

    def clear(self):
        self.index = 0
        self.size = 0

    def generate_batch_data(self, indices):
        states = torch.from_numpy(self.states[indices]).float().to(self.device)
        actions = torch.from_numpy(self.actions[indices]).float().to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).float().to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).float().to(self.device)
        masks = torch.from_numpy(self.masks[indices]).float().to(self.device)
        constraints = torch.from_numpy(self.constraints[indices]).float().to(self.device)
        return {'state': states, 'action': actions, 'next_state': next_states, 'reward': rewards, 'mask': masks, 'constraints': constraints}


class DiscreteOffPolicyBuffer(BaseBuffer):
    def __init__(self, buffer_size, obs_space, action_space, device=torch.device('cpu')):
        super(DiscreteOffPolicyBuffer, self).__init__()
        self.buffer_size = buffer_size
        self.device = device
        self.obs_space =  obs_space
        self.action_space = action_space
        self.obs_shape = obs_space.shape
        self.obs_dtype = obs_space.dtype
        if isinstance(action_space, gym.spaces.discrete.Discrete):
            self.action_dim = 1
            #action_dim = action_space.n
            self.discrete_action = True
        elif isinstance(action_space, gym.spaces.box.Box):
            self.action_dim = action_space.shape[0]
            self.discrete_action = False


        self.states = np.zeros((buffer_size,) + self.obs_shape, dtype=self.obs_dtype)
        self.next_states = np.zeros((buffer_size,) + self.obs_shape, dtype=self.obs_dtype)
        self.action_probs = np.zeros((buffer_size, action_space.n))
        if self.discrete_action:
            self.actions = np.zeros((buffer_size, )).astype(np.int8)
        else:
            self.actions = np.zeros((buffer_size,self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, 1))
        self.masks = np.ones((buffer_size, 1))


        self.index = 0
        self.size = 0

    def to(self, device):
        self.device = device

    def insert(self, state: np.ndarray, action_prob: np.ndarray,  next_state: np.ndarray, reward: np.ndarray,
               mask: np.ndarray):
        self.states[self.index, :] = state.copy()
        self.action_probs[self.index, :] = action_prob.copy()
        self.next_states[self.index, :] = next_state.copy()
        self.rewards[self.index, :] = reward.copy()
        self.masks[self.index, :] = mask.copy()

        self.index = (self.index + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def insert_expert_buffer(self, buffer: ExpertBuffer):
        assert self.index == 0 and self.size == 0 and self.buffer_size > buffer.size
        for i in range(buffer.size):
            self.insert(buffer.states[i], buffer.actions[i], buffer.next_states[i], np.array([0]), np.array([1.0]))

    def clear(self):
        self.index = 0
        self.size = 0

    def generate_batch_data(self, indices):
        states = torch.from_numpy(self.states[indices]).float().to(self.device)
        action_probs = torch.from_numpy(self.action_probs[indices]).float().to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).float().to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).float().to(self.device)
        masks = torch.from_numpy(self.masks[indices]).float().to(self.device)
        return {'state': states, 'action_prob': action_probs, 'next_state': next_states, 'reward': rewards, 'mask': masks}
