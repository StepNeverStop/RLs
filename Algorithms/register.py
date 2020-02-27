import importlib
import tensorflow as tf
from common.yaml_ops import load_yaml
assert tf.__version__[0] == '2'
# algorithms based on TF 2.0
algos = {
    'pg':       {'class': 'PG',     'policy': 'on-policy',  'update': 'perEpisode'},
    'trpo':     {'class': 'TRPO',   'policy': 'on-policy',  'update': 'perEpisode'},
    'ppo':      {'class': 'PPO',    'policy': 'on-policy',  'update': 'perEpisode'},
    'ac':       {'class': 'AC',     'policy': 'off-policy', 'update': 'perStep'},  # could be on-policy, but also doesn't work well.
    'a2c':      {'class': 'A2C',    'policy': 'on-policy',  'update': 'perEpisode'},
    'dpg':      {'class': 'DPG',    'policy': 'off-policy', 'update': 'perStep'},
    'ddpg':     {'class': 'DDPG',   'policy': 'off-policy', 'update': 'perStep'},
    'td3':      {'class': 'TD3',    'policy': 'off-policy', 'update': 'perStep'},
    'sac_v':    {'class': 'SAC_V',  'policy': 'off-policy', 'update': 'perStep'},
    'sac':      {'class': 'SAC',    'policy': 'off-policy', 'update': 'perStep'},
    'tac':      {'class': 'TAC',    'policy': 'off-policy', 'update': 'perStep'},
    'dqn':      {'class': 'DQN',    'policy': 'off-policy', 'update': 'perStep'},
    'drqn':     {'class': 'DRQN',   'policy': 'off-policy', 'update': 'perStep'},
    'drdqn':    {'class': 'DRDQN',  'policy': 'off-policy', 'update': 'perStep'},
    'ddqn':     {'class': 'DDQN',   'policy': 'off-policy', 'update': 'perStep'},
    'dddqn':    {'class': 'DDDQN',  'policy': 'off-policy', 'update': 'perStep'},
    'c51':      {'class': 'C51',    'policy': 'off-policy', 'update': 'perStep'},
    'qrdqn':    {'class': 'QRDQN',  'policy': 'off-policy', 'update': 'perStep'},
    'rainbow':  {'class': 'RAINBOW','policy': 'off-policy', 'update': 'perStep'},
    'iqn':      {'class': 'IQN',    'policy': 'off-policy', 'update': 'perStep'},
    'maxsqn':   {'class': 'MAXSQN', 'policy': 'off-policy', 'update': 'perStep'},
    'ma_dpg':   {'class': 'MADPG',  'policy': 'off-policy', 'update': 'perStep'},
    'ma_ddpg':  {'class': 'MADDPG', 'policy': 'off-policy', 'update': 'perStep'},
    'ma_td3':   {'class': 'MATD3',  'policy': 'off-policy', 'update': 'perStep'}
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
        raise NotImplementedError
    else:
        class_name = algos[name]['class']
        policy_mode = algos[name]['policy']
        model_file = importlib.import_module('Algorithms.tf2algos.' + name)
        model = getattr(model_file, class_name)
        algo_general_config = load_yaml(f'Algorithms/config.yaml')['general']
        algo_config = load_yaml(f'Algorithms/config.yaml')[name]
        algo_config.update(algo_general_config)
        return model, algo_config, policy_mode
