from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import numpy as np
import gym
from torch.distributions import Normal, TransformedDistribution, AffineTransform, TanhTransform, Categorical
from unstable_baselines.common.networks import SequentialNetwork

class EnsembleActor(nn.Module):
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self,  state_dim: int, action_dim: int, network_params: List[int], num_actors: int):
        super(EnsembleActor, self).__init__()
        hidden_dims = [p[1] for p in network_params]
        self.actor_models = [SequentialNetwork(state_dim, action_dim, network_params) for i in range(num_actors)]
        #self.actor_models = [MLP(state_dim, action_dim, hidden_dims, 'ReLU') for i in range(num_actors)]
        self.action_dim = action_dim

        # for actor_model in self.actor_models:
        #     actor_model.init((torch.nn.init.orthogonal_, lambda t: torch.nn.init.constant_(t, 0)),
        #                       (torch.nn.init.orthogonal_, lambda t: torch.nn.init.constant_(t, 0)))

    def forward(self, states):
        actions = [torch.tanh(actor_model(states)) for actor_model in self.actor_models]

        return { 'actions': actions}

    @torch.no_grad()
    def act(self, states):
        means = [torch.tanh(actor_model(states)).cpu().numpy() for actor_model in self.actor_models]

        action_mean = np.mean(means, axis=0)
        action_var = np.var(means, axis=0).mean()
        return { 'mean': action_mean, 'variance': action_var}

    def to(self, device):
        for i, net in enumerate(self.actor_models):
            self.actor_models[i] = self.actor_models[i].to(device)
            
    

class Actor(nn.Module):
    LOG_STD_MIN = -5
    LOG_STD_MAX = 2

    def __init__(self, observation_space, action_dim: int, network_params: List[tuple]):
        super(Actor, self).__init__()

        if len(observation_space.shape) == 1:
            print(observation_space.shape[0], 2 * action_dim)
            self.actor_model = SequentialNetwork(observation_space.shape[0], 2 * action_dim, network_params, act_fn='relu', out_act_fn='identity')
        elif len(observation_space.shape) == 3:
            self.actor_model = SequentialNetwork(observation_space.shape[0], 2 * action_dim, network_params, act_fn='relu', out_act_fn='identity')
        self.action_dim = action_dim

        # self.actor_model.init((torch.nn.init.orthogonal_, lambda t: torch.nn.init.constant_(t, 0)),
        #                       (torch.nn.init.orthogonal_, lambda t: torch.nn.init.constant_(t, 0)))

    def forward(self, states):
        means, log_stds = torch.split(self.actor_model(states), self.action_dim, dim=-1)
        log_stds = torch.tanh(log_stds)
        log_stds = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_stds + 1)
        stds = torch.exp(log_stds)

        dists = TransformedDistribution(Normal(torch.zeros_like(means), 1.0),
                                        [AffineTransform(means, stds, event_dim=1), TanhTransform()]
                                        )
        samples = dists.rsample()
        log_probs = dists.log_prob(samples)
        means = torch.tanh(means)

        return {'sample': samples, 'mean': means, 'log_std': log_stds, 'log_prob': log_probs, 'dist': dists}



    
class DiscreteActor(nn.Module):

    def __init__(self, observation_space, action_dim: int, network_params: str):
        super(DiscreteActor, self).__init__()
        if len(observation_space.shape) == 1:
            self.actor_model = SequentialNetwork(observation_space.shape[0], action_dim, network_params, act_fn='relu', out_act_fn='identity')
        elif len(observation_space.shape) == 3:
            self.actor_model = SequentialNetwork(observation_space.shape, action_dim, network_params, act_fn='relu', out_act_fn='identity')
        self.action_dim = action_dim

        #self.actor_model.init((torch.nn.init.orthogonal_, lambda t: torch.nn.init.constant_(t, 0)),
        #                      (torch.nn.init.orthogonal_, lambda t: torch.nn.init.constant_(t, 0)))

    def forward(self, states):
        if len(states.shape) == 4:
            states = states / 255.0
        logits = self.actor_model(states)
        probs = torch.softmax(logits, dim=-1)

        dists = Categorical(probs=probs)

        samples = dists.sample()
        log_probs = dists.log_prob(samples)

        modes = torch.argmax(probs, dim=-1, keepdim=True)

        return {'sample': samples, 'mode': modes, 'prob': probs,
                'log_prob': log_probs, 'dist': dists}