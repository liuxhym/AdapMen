from __future__ import annotations
from cmath import exp

import copy
from typing import Dict
from operator import itemgetter

from munch import Munch
import torch
import torch.nn as nn
import math

class BC:
    def __init__(self, actor: nn.Module, cfg: Munch):
        self.actor = actor
        self.cfg = copy.deepcopy(cfg)

        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=cfg.bc.actor_lr)

    def update(self, expert_data_batch: Dict[str, torch.Tensor]):
        self.actor.train()

        expert_states, expert_actions = itemgetter('state', 'action')(expert_data_batch)
        actor_actions, actor_logprobs, actor_log_stds = itemgetter('mean', 'log_prob', 'log_std')(self.actor(expert_states))

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(expert_actions, actor_actions)
        # elif self.bc_type == "mle":
        #     actor_stds = actor_log_stds.exp()
        #     losses = torch.log(2*math.pi*(actor_stds**2))/2 + ((actor_actions - expert_actions)**2) / (2 * actor_stds**2)
        #     loss = losses.sum(axis=1).mean()
        # else:
        #     raise NotImplementedError
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            avg_pi_entropy = (-actor_logprobs).mean().cpu().item()

        return {'loss/actor_loss': loss.item(), 'loss/avg_pi_entropy': avg_pi_entropy}


class DiscreteBC:
    def __init__(self, actor: nn.Module, cfg: Munch):
        self.actor = actor
        self.cfg = copy.deepcopy(cfg)

        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=cfg.bc.actor_lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def update(self, expert_data_batch: Dict[str, torch.Tensor]):

        expert_states, expert_actions = itemgetter('state', 'action')(expert_data_batch)
        actor_probs = itemgetter('prob')(self.actor(expert_states))
        loss = self.loss_fn(actor_probs, expert_actions.long())

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {'loss/actor_loss': loss.item()}
