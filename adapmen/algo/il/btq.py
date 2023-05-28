from __future__ import annotations
from adapmen.algo.il.bc import BC, DiscreteBC

import copy

from munch import Munch
import torch
import torch.nn as nn


class BTQ(BC):
    def __init__(self, actor: nn.Module, cfg: Munch):
        
        self.actor = actor
        self.cfg = copy.deepcopy(cfg)

        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=cfg.btq.actor_lr)
        self.bc_type = cfg.btq.bc_type
        
class DiscreteBTQ(DiscreteBC):
    def __init__(self, actor: nn.Module, cfg: Munch):
        
        self.actor = actor
        self.cfg = copy.deepcopy(cfg)

        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=cfg.btq.actor_lr)
        self.loss_fn = torch.nn.CrossEntropyLoss()
    