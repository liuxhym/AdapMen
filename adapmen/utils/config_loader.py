from __future__ import annotations
from distutils.command.config import config
from typing import List, Union, Tuple, Dict
import os

import argparse
import ast
import munch
from munch import Munch
import collections
from yaml import load
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from unstable_baselines.common.util import load_config

def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, safe_eval(v)))
    return dict(items)


def deflatten(d, sep='.'):
    deflattend_d = {}
    for k, v in d.items():
        d = deflattend_d
        key_seq = k.split(sep)
        for key in key_seq[:-1]:
            try:
                d = d[key]
            except (TypeError, KeyError):
                d[key] = {}
                d = d[key]
        d[key_seq[-1]] = v
    return deflattend_d


def safe_eval(x: str):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return x


def change_dict_value(obj, key_sequence: List[str], value):
    try:
        for key in key_sequence[:-1]:
            obj = obj[key]
        obj[key_sequence[-1]] = value
    except KeyError:
        raise KeyError('Incorrect key sequences.')

class ConfigLoader:
    def __new__(cls, config_paths: Union[List[str], str] = 'config.yaml') -> Tuple[Munch, Dict]:
        parser = argparse.ArgumentParser(description='Mahjong AI')
        parser.add_argument('-c', '--configs', type=str, help='configuration file (YAML)', nargs='+', action='append')
        parser.add_argument('-s', '--set', type=str, help='additional options to override configuration',
                            nargs='*', action='append')

        args, unknown = parser.parse_known_args()
        config_dict = {}
        flat_config_dict = {}
        overwritten_config_dict = {}

        if args.configs:
            config_paths = args.configs
            flat_config_paths = [item for sublist in config_paths for item in sublist]
        else:
            if isinstance(config_paths, str):
                config_paths = [config_paths]
            flat_config_paths = config_paths

        assert isinstance(flat_config_paths, list)

        for config_path in flat_config_paths:
            assert isinstance(config_path, str)
            if not config_path.startswith('/') and not os.path.exists(config_path):
                config_path = os.path.join(os.path.dirname(__file__), config_path)

            print('Info: Loading configs from {}.'.format(os.path.abspath(config_path)))

            with open(config_path, 'r', encoding='utf-8') as f:
                print(config_path)
                new_config_dict = load(f, Loader=Loader)
                flat_new_config_dict = flatten(new_config_dict)
                overwritten_config_dict.update({k: v for k, v in flat_new_config_dict.items()
                                                if (k in flat_config_dict.keys() and v != flat_config_dict[k])})
                flat_config_dict.update(flat_new_config_dict)
                config_dict = deflatten(flat_config_dict)

        if args.set:
            for instruction in sum(args.set, []):
                key, value = instruction.split('=')
                change_dict_value(config_dict, key.split('.'), ast.literal_eval(value))
                overwritten_config_dict.update({key: ast.literal_eval(value)})

        for key, value in overwritten_config_dict.items():
            print('Info: Hyperparams {} has been overwritten to {}.'.format(key, value))

        config = munch.munchify(config_dict)
        config_dict = flatten(config_dict)
        logged_config_dict = {}

        for key, value in config_dict.items():
            if key.find('.') >= 0:
                logged_config_dict[key] = value
        return config, logged_config_dict