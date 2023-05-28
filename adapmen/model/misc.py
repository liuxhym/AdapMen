from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def soft_update(source_model: nn.Module, target_model: nn.Module, tau):
    for target_param, param in zip(target_model.parameters(), source_model.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def inv_grad(net):
    for param in net.parameters():
        param.grad *= -1


def orthogonal_regularization(model, reg_coeff=1e-4):
    orth_loss = 0.0
    for name, param in model.named_parameters():
        if 'bias' not in name and len(param.shape)<=2:
            prod = torch.mm(torch.t(param), param)
            orth_loss += (torch.square(prod * (1 - torch.eye(prod.shape[0], device=prod.device)))).sum()
    return orth_loss * reg_coeff


def init(module, weight_init=None, bias_init=None):
    if weight_init:
        weight_init(module.weight.data)
    if bias_init:
        bias_init(module.bias.data)


def get_flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad