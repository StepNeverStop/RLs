import importlib
import tensorflow as tf
from common.yaml_ops import load_yaml
assert tf.__version__[0] == '2'
# algorithms based on TF 2.0
algos = {
    # On-Policy
    'pg':       {'class': 'PG',     'policy': 'on-policy',  'update': 'perEpisode'},
    'trpo':     {'class': 'TRPO',   'policy': 'on-policy',  'update': 'perEpisode'},
    'ppo':      {'class': 'PPO',    'policy': 'on-policy',  'update': 'perEpisode'},
    'a2c':      {'class': 'A2C',    'policy': 'on-policy',  'update': 'perEpisode'},
    'cem':      {'class': 'CEM',    'policy': 'on-policy',  'update': 'perEpisode'},
    'aoc':      {'class': 'AOC',    'policy': 'on-policy',  'update': 'perEpisode'},
    'ppoc':     {'class': 'PPOC',   'policy': 'on-policy',  'update': 'perEpisode'},

    # Off-Policy
    'ac':       {'class': 'AC',     'policy': 'off-policy', 'update': 'perStep'},
    'dpg':      {'class': 'DPG',    'policy': 'off-policy', 'update': 'perStep'},
    'ddpg':     {'class': 'DDPG',   'policy': 'off-policy', 'update': 'perStep'},
    'pd_ddpg':  {'class': 'PD_DDPG','policy': 'off-policy', 'update': 'perStep'},
    'td3':      {'class': 'TD3',    'policy': 'off-policy', 'update': 'perStep'},
    'sac_v':    {'class': 'SAC_V',  'policy': 'off-policy', 'update': 'perStep'},
    'sac':      {'class': 'SAC',    'policy': 'off-policy', 'update': 'perStep'},
    'tac':      {'class': 'TAC',    'policy': 'off-policy', 'update': 'perStep'},
    'dqn':      {'class': 'DQN',    'policy': 'off-policy', 'update': 'perStep'},
    'ddqn':     {'class': 'DDQN',   'policy': 'off-policy', 'update': 'perStep'},
    'dddqn':    {'class': 'DDDQN',  'policy': 'off-policy', 'update': 'perStep'},
    'c51':      {'class': 'C51',    'policy': 'off-policy', 'update': 'perStep'},
    'qrdqn':    {'class': 'QRDQN',  'policy': 'off-policy', 'update': 'perStep'},
    'rainbow':  {'class': 'RAINBOW','policy': 'off-policy', 'update': 'perStep'},
    'iqn':      {'class': 'IQN',    'policy': 'off-policy', 'update': 'perStep'},
    'maxsqn':   {'class': 'MAXSQN', 'policy': 'off-policy', 'update': 'perStep'},
    'ma_ddpg':  {'class': 'MADDPG', 'policy': 'off-policy', 'update': 'perStep'},
    'sql':      {'class': 'SQL',    'policy': 'off-policy', 'update': 'perStep'},
    'bootstrappeddqn': {'class': 'BootstrappedDQN',    'policy': 'off-policy', 'update': 'perStep'},
    'oc':       {'class': 'OC',     'policy': 'off-policy', 'update': 'perStep'},
    'ioc':      {'class': 'IOC',    'policy': 'off-policy', 'update': 'perStep'},
    'qs':       {'class': 'QS',     'policy': 'off-policy', 'update': 'perStep'},
    'hiro':     {'class': 'HIRO',   'policy': 'off-policy', 'update': 'perStep'},
    'curl':     {'class': 'CURL',   'policy': 'off-policy', 'update': 'perStep'},
}


def get_model_info(name: str):
    '''
    Args:
        name: name of algorithms
    Return:
        class of the algorithm model named `name`.
        defaulf config of specified algorithm.
        mode of policy, `on-policy` or `off-policy`
    '''
    if name not in algos.keys():
        raise NotImplementedError(name)
    else:
        class_name = algos[name]['class']
        policy_mode = algos[name]['policy']
        model_file = importlib.import_module('algos.tf2algos.' + name)
        model = getattr(model_file, class_name)
        algo_general_config = load_yaml(f'algos/config.yaml')['general']
        if policy_mode == 'on-policy':
            algo_policy_config = load_yaml(f'algos/config.yaml')['on_policy']
        elif policy_mode == 'off-policy':
            algo_policy_config = load_yaml(f'algos/config.yaml')['off_policy']
        algo_config = load_yaml(f'algos/config.yaml')[name]
        algo_config.update(algo_policy_config)
        algo_config.update(algo_general_config)
        return model, algo_config, policy_mode
