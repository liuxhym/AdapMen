import random

import numpy as np
import torch


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def dict_to_list(log_infos: dict):
    infos = []
    for name, value in log_infos.items():
        infos.append((name, value))
    return infos
