from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


from unstable_baselines.common.networks import SequentialNetwork


class Critic(nn.Module):
    def __int__(self, state_dim: int, hidden_dims: List[int]):
        super(Critic, self).__int__()
        network_params = [("mlp", d) for d in hidden_dims]
        self.critic_model1 = SequentialNetwork(state_dim, 1, network_params, 'ReLU')
        self.critic_model2 = SequentialNetwork(state_dim, 1, network_params, 'ReLU')

        self.critic_model1.init((torch.nn.init.orthogonal_, lambda t: torch.nn.init.constant_(t, 0)),
                                (torch.nn.init.orthogonal_, lambda t: torch.nn.init.constant_(t, 0)))
        self.critic_model2.init((torch.nn.init.orthogonal_, lambda t: torch.nn.init.constant_(t, 0)),
                                (torch.nn.init.orthogonal_, lambda t: torch.nn.init.constant_(t, 0)))

    def forward(self, states, actions):
        x = torch.cat([states, actions], -1)

        q1 = self.critic_model1(x)
        q2 = self.critic_model2(x)

        return q1, q2
