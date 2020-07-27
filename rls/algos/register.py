#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import importlib
import tensorflow as tf

from rls.common.yaml_ops import load_yaml
assert tf.__version__[0] == '2'
# algorithms based on TF 2.0
algos = {
    # On-Policy
    'pg':       {'class': 'PG',     'policy': 'on-policy',  'update': 'perEpisode', 'type': 'single'},
    'trpo':     {'class': 'TRPO',   'policy': 'on-policy',  'update': 'perEpisode', 'type': 'single'},
    'ppo':      {'class': 'PPO',    'policy': 'on-policy',  'update': 'perEpisode', 'type': 'single'},
    'a2c':      {'class': 'A2C',    'policy': 'on-policy',  'update': 'perEpisode', 'type': 'single'},
    'cem':      {'class': 'CEM',    'policy': 'on-policy',  'update': 'perEpisode', 'type': 'single'},
    'aoc':      {'class': 'AOC',    'policy': 'on-policy',  'update': 'perEpisode', 'type': 'hierarchical'},
    'ppoc':     {'class': 'PPOC',   'policy': 'on-policy',  'update': 'perEpisode', 'type': 'hierarchical'},

    # Off-Policy
    'qs':       {'class': 'QS',     'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'ac':       {'class': 'AC',     'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'dpg':      {'class': 'DPG',    'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'ddpg':     {'class': 'DDPG',   'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'pd_ddpg':  {'class': 'PD_DDPG','policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'td3':      {'class': 'TD3',    'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'sac_v':    {'class': 'SAC_V',  'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'sac':      {'class': 'SAC',    'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'tac':      {'class': 'TAC',    'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'dqn':      {'class': 'DQN',    'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'ddqn':     {'class': 'DDQN',   'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'dddqn':    {'class': 'DDDQN',  'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'c51':      {'class': 'C51',    'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'qrdqn':    {'class': 'QRDQN',  'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'rainbow':  {'class': 'RAINBOW','policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'iqn':      {'class': 'IQN',    'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'maxsqn':   {'class': 'MAXSQN', 'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'sql':      {'class': 'SQL',    'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'bootstrappeddqn': {'class': 'BootstrappedDQN',    'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'curl':     {'class': 'CURL',   'policy': 'off-policy', 'update': 'perStep', 'type': 'single'},
    'oc':       {'class': 'OC',     'policy': 'off-policy', 'update': 'perStep', 'type': 'hierarchical'},
    'ioc':      {'class': 'IOC',    'policy': 'off-policy', 'update': 'perStep', 'type': 'hierarchical'},
    'hiro':     {'class': 'HIRO',   'policy': 'off-policy', 'update': 'perStep', 'type': 'hierarchical'},
    'ma_ddpg':  {'class': 'MADDPG', 'policy': 'off-policy', 'update': 'perStep', 'type': 'multi'},
}


def get_model_info(name: str):
    '''
    Args:
        name: name of algorithms
    Return:
        class of the algorithm model named `name`.
        defaulf config of specified algorithm.
        type of policy, `on-policy` or `off-policy`
    '''
    if name not in algos.keys():
        raise NotImplementedError(name)
    else:
        class_name = algos[name]['class']
        policy_mode = algos[name]['policy']
        _type = algos[name]['type']

        model_file = importlib.import_module(f'rls.algos.{_type}.{name}')
        model = getattr(model_file, class_name)
        algo_general_config = load_yaml(f'rls/algos/config.yaml')['general']
        if policy_mode == 'on-policy':
            algo_policy_config = load_yaml(f'rls/algos/config.yaml')['on_policy']
        elif policy_mode == 'off-policy':
            algo_policy_config = load_yaml(f'rls/algos/config.yaml')['off_policy']
        algo_config = load_yaml(f'rls/algos/config.yaml')[name]
        algo_config.update(algo_policy_config)
        algo_config.update(algo_general_config)
        return model, algo_config, policy_mode
