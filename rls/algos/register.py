#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import tensorflow as tf
assert tf.__version__[0] == '2'

# algorithms based on TF 2.x

from typing import \
    Tuple, \
    Callable, \
    Dict

from rls.common.yaml_ops import load_yaml


algos = {
    # On-Policy
    'pg':       {'algo_class': 'PG',     'policy_mode': 'on-policy',  'update_mode': 'perEpisode', 'policy_type': 'single'},
    'trpo':     {'algo_class': 'TRPO',   'policy_mode': 'on-policy',  'update_mode': 'perEpisode', 'policy_type': 'single'},
    'ppo':      {'algo_class': 'PPO',    'policy_mode': 'on-policy',  'update_mode': 'perEpisode', 'policy_type': 'single'},
    'a2c':      {'algo_class': 'A2C',    'policy_mode': 'on-policy',  'update_mode': 'perEpisode', 'policy_type': 'single'},
    'cem':      {'algo_class': 'CEM',    'policy_mode': 'on-policy',  'update_mode': 'perEpisode', 'policy_type': 'single'},
    'aoc':      {'algo_class': 'AOC',    'policy_mode': 'on-policy',  'update_mode': 'perEpisode', 'policy_type': 'hierarchical'},
    'ppoc':     {'algo_class': 'PPOC',   'policy_mode': 'on-policy',  'update_mode': 'perEpisode', 'policy_type': 'hierarchical'},

    # Off-Policy
    'qs':       {'algo_class': 'QS',     'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'ac':       {'algo_class': 'AC',     'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'dpg':      {'algo_class': 'DPG',    'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'ddpg':     {'algo_class': 'DDPG',   'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'pd_ddpg':  {'algo_class': 'PD_DDPG','policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'td3':      {'algo_class': 'TD3',    'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'sac_v':    {'algo_class': 'SAC_V',  'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'sac':      {'algo_class': 'SAC',    'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'tac':      {'algo_class': 'TAC',    'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'dqn':      {'algo_class': 'DQN',    'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'ddqn':     {'algo_class': 'DDQN',   'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'dddqn':    {'algo_class': 'DDDQN',  'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'c51':      {'algo_class': 'C51',    'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'qrdqn':    {'algo_class': 'QRDQN',  'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'rainbow':  {'algo_class': 'RAINBOW','policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'iqn':      {'algo_class': 'IQN',    'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'maxsqn':   {'algo_class': 'MAXSQN', 'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'sql':      {'algo_class': 'SQL',    'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'bootstrappeddqn': {'algo_class': 'BootstrappedDQN',    'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'curl':     {'algo_class': 'CURL',   'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'single'},
    'oc':       {'algo_class': 'OC',     'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'hierarchical'},
    'ioc':      {'algo_class': 'IOC',    'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'hierarchical'},
    'hiro':     {'algo_class': 'HIRO',   'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'hierarchical'},
    'maddpg':   {'algo_class': 'MADDPG', 'policy_mode': 'off-policy', 'update_mode': 'perStep', 'policy_type': 'multi'},
}


def get_model_info(name: str) -> Tuple[Callable, Dict, str, str]:
    '''
    Args:
        name: name of algorithms
    Return:
        algo_class of the algorithm model named `name`.
        defaulf config of specified algorithm.
        policy_type of policy, `on-policy` or `off-policy`
    '''
    if name not in algos.keys():
        raise NotImplementedError(name)
    else:
        class_name = algos[name]['algo_class']
        policy_mode = algos[name]['policy_mode']
        policy_type = algos[name]['policy_type']

        model = getattr(
            importlib.import_module(f'rls.algos.{policy_type}.{name}'), 
            class_name)

        algo_config = load_yaml(f'rls/algos/config.yaml')[name]
        algo_config.update(
            load_yaml(f'rls/algos/config.yaml')[policy_mode.replace('-', '_')]
        )
        algo_config.update(
            load_yaml(f'rls/algos/config.yaml')['general']
        )
        return model, algo_config, policy_mode, policy_type
