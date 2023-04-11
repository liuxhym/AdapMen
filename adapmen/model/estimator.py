from __future__ import annotations

from typing import List

from gym.spaces import Space, Box, Discrete
import torch
import torch.nn as nn
from unstable_baselines.common.networks import SequentialNetwork


class Estimator(nn.Module):
    def __init__(self, observation_space: int, action_space: Space, nu_network_params: List[tuple]):
        super(Estimator, self).__init__()

        use_biases = [True] * len(nu_network_params) + [False]
        if isinstance(action_space, Box):
            action_dim = action_space.shape[0]
            self.estimator_model = SequentialNetwork(observation_space.shape[0] + action_dim, 1, nu_network_params, act_fn='relu', out_act_fn='identity')
        elif isinstance(action_space, Discrete):
            #atari case
            action_dim = action_space.n
            self.estimator_model = SequentialNetwork(observation_space.shape, action_space.n,nu_network_params, act_fn='relu', out_act_fn='identity')
        else:
            raise RuntimeError(f'Unsupported action space: {action_space}')

        for name, param in self.estimator_model.named_parameters():
            if 'bias' not in name and len(param.shape)<=2:
                nn.init.orthogonal_(param)

    def forward(self, states_or_state_actions: torch.Tensor):
        return self.estimator_model(states_or_state_actions)

