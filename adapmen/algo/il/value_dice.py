from __future__ import annotations

import abc
import copy
from operator import itemgetter
from typing import Dict

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from adapmen.model.misc import orthogonal_regularization, inv_grad


EPS = np.finfo(np.float32).eps


def weighted_softmax(x, weights, dim=0):
    x = x - x.max(dim=0)[0]
    return weights * torch.exp(x) / (weights * torch.exp(x)).sum(dim=dim, keepdims=True)


def compute_value_dice_loss(expert_init_nus, expert_cur_nus, expert_next_nus, buffer_cur_nus, buffer_next_nus,
                            discount, replay_reg_coeff):
    expert_diffs = expert_cur_nus - discount * expert_next_nus
    buffer_diffs = buffer_cur_nus - discount * buffer_next_nus

    expert_linear_loss = (expert_init_nus * (1 - discount)).mean()
    buffer_linear_loss = buffer_diffs.mean()

    diffs = torch.concat([expert_diffs, buffer_diffs], dim=0)
    weights = torch.concat([torch.ones_like(expert_diffs) * (1 - replay_reg_coeff),
                            torch.ones_like(buffer_diffs) * replay_reg_coeff
                            ], dim=0)
    weights = weights / weights.sum()

    nonlinear_loss = (weighted_softmax(diffs, weights, dim=0).detach() * diffs).sum()
    linear_loss = expert_linear_loss * (1 - replay_reg_coeff) + buffer_linear_loss * replay_reg_coeff

    loss = nonlinear_loss - linear_loss
    return loss


class ContinuousValueDice:
    def __init__(self, actor: nn.Module, nu_net: nn.Module, cfg: Munch):
        self.cfg = copy.deepcopy(cfg)
        self.nu_net = nu_net
        self.actor = actor

        self.nu_optimizer = torch.optim.Adam(params=self.nu_net.parameters(), lr=cfg.value_dice.nu_lr)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=cfg.value_dice.actor_lr)

    def update(self, expert_data_batch: Dict[str, torch.Tensor], buffer_data_batch: Dict[str, torch.Tensor]) \
            -> Dict[str, float]:
        self.actor.train()
        self.nu_net.train()

        discount = self.cfg.env.discount
        replay_reg_coeff = self.cfg.value_dice.replay_reg_coeff

        expert_states, expert_actions, expert_next_states = \
            itemgetter('state', 'action', 'next_state')(expert_data_batch)

        buffer_states, buffer_actions, buffer_next_states =\
            itemgetter('state', 'action', 'next_state')(buffer_data_batch)

        expert_initial_states = expert_states

        actor_next_expert_actions = itemgetter('sample')(self.actor(expert_next_states))
        actor_next_buffer_actions, actor_next_buffer_logprobs = \
            itemgetter('sample', 'log_prob')(self.actor(buffer_next_states))
        actor_initial_expert_actions = itemgetter('sample')(self.actor(expert_initial_states))

        expert_init_inputs = torch.concat([expert_initial_states, actor_initial_expert_actions], dim=-1)
        expert_cur_inputs = torch.concat([expert_states, expert_actions], dim=-1)
        expert_next_inputs = torch.concat([expert_next_states, actor_next_expert_actions], dim=-1)

        buffer_cur_inputs = torch.concat([buffer_states, buffer_actions], dim=-1)
        buffer_next_inputs = torch.concat([buffer_next_states, actor_next_buffer_actions], dim=-1)

        expert_init_nus = self.nu_net(expert_init_inputs)
        expert_cur_nus = self.nu_net(expert_cur_inputs)
        expert_next_nus = self.nu_net(expert_next_inputs)

        buffer_cur_nus = self.nu_net(buffer_cur_inputs)
        buffer_next_nus = self.nu_net(buffer_next_inputs)

        loss = compute_value_dice_loss(expert_init_nus, expert_cur_nus, expert_next_nus,
                                       buffer_cur_nus, buffer_next_nus, discount, replay_reg_coeff)

        alpha = torch.rand(size=(expert_cur_inputs.shape[0], 1)).to(expert_cur_inputs.device)

        interpolated_cur_nu_inputs = alpha * expert_cur_inputs + (1 - alpha) * buffer_cur_inputs
        interpolated_cur_next_nu_inputs = alpha * expert_next_inputs + (1 - alpha) * buffer_next_inputs

        interpolated_nu_inputs = torch.concat([interpolated_cur_nu_inputs, interpolated_cur_next_nu_inputs], dim=0)
        interpolated_nu_inputs.requires_grad_(True)

        with torch.enable_grad():
            interpolated_nus = self.nu_net(interpolated_nu_inputs)
            nu_grads = torch.autograd.grad(outputs=interpolated_nus, inputs=interpolated_nu_inputs,
                                           grad_outputs=torch.ones_like(interpolated_nus),
                                           create_graph=True)[0] + EPS
            nu_grad_penalty = nu_grads.norm(2, dim=-1).sub(1).pow(2).mean()

        self.nu_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        (loss + nu_grad_penalty * self.cfg.value_dice.nu_reg_coeff - orthogonal_regularization(self.actor)).backward()
        inv_grad(self.actor)
        self.actor_optimizer.step()
        self.nu_optimizer.step()

        with torch.no_grad():
            avg_expert_nu = expert_cur_nus.mean().cpu().item()
            avg_buffer_nu = buffer_cur_nus.mean().cpu().item()

            avg_nu_grad_penalty = nu_grad_penalty.mean().cpu().item()
            avg_nu_loss = (loss + nu_grad_penalty * self.cfg.value_dice.nu_reg_coeff).mean().cpu().item()

            avg_pi_loss = (-loss + orthogonal_regularization(self.actor)).mean().cpu().item()
            avg_pi_entropy = (-actor_next_buffer_logprobs).mean().cpu().item()

        return {'train/avg_expert_nu': avg_expert_nu, 'train/avg_buffer_nu': avg_buffer_nu,
                'train/avg_nu_grad_penalty': avg_nu_grad_penalty, 'loss/avg_nu_loss': avg_nu_loss,
                'loss/avg_pi_loss': avg_pi_loss, 'loss/avg_pi_entropy': avg_pi_entropy}


class DiscreteValueDice:
    def __init__(self, actor: nn.Module, nu_net: nn.Module, cfg: Munch):
        self.cfg = copy.deepcopy(cfg)
        self.nu_net = nu_net
        self.actor = actor

        self.nu_optimizer = torch.optim.Adam(params=self.nu_net.parameters(), lr=cfg.value_dice.nu_lr)
        self.actor_optimizer = torch.optim.Adam(params=self.actor.parameters(), lr=cfg.value_dice.actor_lr)

    def update(self, expert_data_batch: Dict[str, torch.Tensor], buffer_data_batch: Dict[str, torch.Tensor]) \
            -> Dict[str, float]:
        self.actor.train()
        self.nu_net.train()

        discount = self.cfg.env.discount
        replay_reg_coeff = self.cfg.value_dice.replay_reg_coeff

        expert_states, expert_actions, expert_next_states = \
            itemgetter('state', 'action', 'next_state')(expert_data_batch)
        expert_actions = expert_actions.long()

        buffer_states, buffer_action_probs, buffer_next_states =\
            itemgetter('state', 'action_prob', 'next_state')(buffer_data_batch)
        expert_initial_states = expert_states

        actor_next_expert_action_probs = (self.actor(expert_next_states))['prob']
        actor_next_buffer_action_probs = (self.actor(buffer_next_states))['prob']
        actor_initial_expert_action_probs = self.actor(expert_initial_states)['prob']

        expert_init_nus = torch.bmm(self.nu_net(expert_initial_states).unsqueeze(1),
                                    actor_initial_expert_action_probs.unsqueeze(2)).squeeze(-1)
        expert_cur_nus = self.nu_net(expert_states).gather(1, expert_actions.unsqueeze(1))
        expert_next_nus = torch.bmm(self.nu_net(expert_next_states).unsqueeze(1),
                                    actor_next_expert_action_probs.unsqueeze(2)).squeeze(-1)

        buffer_cur_nus = torch.bmm(self.nu_net(buffer_states).unsqueeze(1),
                                   buffer_action_probs.unsqueeze(2)).squeeze(-1)
        buffer_next_nus = torch.bmm(self.nu_net(buffer_next_states).unsqueeze(1),
                                    actor_next_buffer_action_probs.unsqueeze(2)).squeeze(-1)

        loss = compute_value_dice_loss(expert_init_nus, expert_cur_nus, expert_next_nus,
                                       buffer_cur_nus, buffer_next_nus, discount, replay_reg_coeff)

        alpha = torch.rand(size=(expert_states.shape[0], 1)).to(expert_states.device)
        # interpolated_cur_states = alpha * expert_states + (1 - alpha) * buffer_states
        # interpolated_next_states = alpha * expert_next_states + (1 - alpha) * buffer_next_states
        interpolated_cur_states = torch.stack([a * es + (1-a) * bs for a, es, bs in zip(alpha, expert_states, buffer_states)])
        interpolated_next_states = torch.stack([a * ens + (1-a) * bns for a, ens, bns in zip(alpha, expert_next_states, buffer_next_states)])
        # interpolated_next_states = alpha * expert_next_states + (1 - alpha) * buffer_next_states

        expert_action_probs = functional.one_hot(expert_actions, num_classes=buffer_action_probs.shape[-1])
        interpolated_cur_action_probs = \
            alpha * expert_action_probs + (1 - alpha) * buffer_action_probs
        interpolated_next_action_probs = \
            alpha * actor_next_expert_action_probs + (1 - alpha) * actor_next_buffer_action_probs

        interpolated_nu_states = torch.concat([interpolated_cur_states, interpolated_next_states], dim=0)
        interpolated_action_probs = \
            torch.concat([interpolated_cur_action_probs, interpolated_next_action_probs], dim=0)
        interpolated_nu_states.requires_grad_(True)
        interpolated_action_probs.requires_grad_(True)

        with torch.enable_grad():
            interpolated_nus = torch.bmm(self.nu_net(interpolated_nu_states).unsqueeze(1),
                                         interpolated_action_probs.unsqueeze(2))
            nu_grads = torch.autograd.grad(outputs=interpolated_nus, inputs=interpolated_nu_states,
                                           grad_outputs=torch.ones_like(interpolated_nus),
                                           create_graph=True)[0] + EPS
            nu_grad_penalty = nu_grads.norm(2, dim=-1).sub(1).pow(2).mean()

        self.nu_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        (loss + nu_grad_penalty * self.cfg.value_dice.nu_reg_coeff - orthogonal_regularization(self.actor)).backward()
        inv_grad(self.actor)
        self.actor_optimizer.step()
        self.nu_optimizer.step()

        with torch.no_grad():
            avg_expert_nu = expert_cur_nus.mean().cpu().item()
            avg_buffer_nu = buffer_cur_nus.mean().cpu().item()

            avg_nu_grad_penalty = nu_grad_penalty.mean().cpu().item()
            avg_nu_loss = (loss + nu_grad_penalty * self.cfg.value_dice.nu_reg_coeff).mean().cpu().item()

            avg_pi_loss = (-loss + orthogonal_regularization(self.actor)).mean().cpu().item()

        return {'train/avg_expert_nu': avg_expert_nu, 'train/avg_buffer_nu': avg_buffer_nu,
                'train/avg_nu_grad_penalty': avg_nu_grad_penalty, 'loss/avg_nu_loss': avg_nu_loss,
                'loss/avg_pi_loss': avg_pi_loss}
