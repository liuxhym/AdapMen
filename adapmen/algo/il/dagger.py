from __future__ import annotations
from adapmen.algo.il.bc import BC, DiscreteBC

import copy
from typing import Dict

from munch import Munch
import torch
import torch.nn as nn
import torch.nn.functional as F
from operator import itemgetter


class DAgger(BC):
    def __init__(self, actor: nn.Module, cfg: Munch):
        self.actor = actor
        self.cfg = copy.deepcopy(cfg)

        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=cfg.dagger.actor_lr)
        self.loss_fn = torch.nn.MSELoss()
        self.bc_type = "mse"

class EnsembleDAgger():
    def __init__(self, actor: nn.Module, cfg: Munch):
        self.actor = actor
        self.cfg = copy.deepcopy(cfg)

        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=cfg.bc.actor_lr)
        self.loss_fn = torch.nn.MSELoss()

    def update(self, expert_data_batch: Dict[str, torch.Tensor]):
        self.actor.train()

        expert_states, expert_actions = itemgetter('state', 'action')(expert_data_batch)
        actor_actions, actor_logprobs = itemgetter('mean', 'log_prob')(self.actor(expert_states))
        loss = self.loss_fn(expert_actions, actor_actions)

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            avg_pi_entropy = (-actor_logprobs).mean().cpu().item()

        return {'loss/actor_loss': loss.item(), 'loss/avg_pi_entropy': avg_pi_entropy}
        
class DiscreteDAgger(DiscreteBC):
    def __init__(self, actor: nn.Module, cfg: Munch):
        self.actor = actor
        self.cfg = copy.deepcopy(cfg)

        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=cfg.dagger.actor_lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()