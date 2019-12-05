import importlib
import tensorflow as tf
assert tf.__version__[0] == '2'
# algorithms based on TF 2.0
algos = {
    'pg': {
        'class': 'PG',
        'policy': 'on-policy',
        'update': 'perEpisode'
    },
    'ppo': {
        'class': 'PPO',
        'policy': 'on-policy',
        'update': 'perEpisode'
    },
    'ac': {
        'class': 'AC',
        'policy': 'off-policy',
        'update': 'perStep'
    },  # could be on-policy, but also doesn't work well.
    'a2c': {
        'class': 'A2C',
        'policy': 'on-policy',
        'update': 'perEpisode'
    },
    'dpg': {
        'class': 'DPG',
        'policy': 'off-policy',
        'update': 'perStep'
    },
    'ddpg': {
        'class': 'DDPG',
        'policy': 'off-policy',
        'update': 'perStep'
    },
    'td3': {
        'class': 'TD3',
        'policy': 'off-policy',
        'update': 'perStep'
    },
    'sac_v': {
        'class': 'SAC_V',
        'policy': 'off-policy',
        'update': 'perStep'
    },
    'sac': {
        'class': 'SAC',
        'policy': 'off-policy',
        'update': 'perStep'
    },
    'tac': {
        'class': 'TAC',
        'policy': 'off-policy',
        'update': 'perStep'
    },
    'dqn': {
        'class': 'DQN',
        'policy': 'off-policy',
        'update': 'perStep'
    },
    'ddqn': {
        'class': 'DDQN',
        'policy': 'off-policy',
        'update': 'perStep'
    },
    'dddqn': {
        'class': 'DDDQN',
        'policy': 'off-policy',
        'update': 'perStep'
    },
    'maxsqn': {
        'class': 'MAXSQN',
        'policy': 'off-policy',
        'update': 'perStep'
    },
    'ma_dpg': {
        'class': 'MADPG',
        'policy': 'off-policy',
        'update': 'perStep'
    },
    'ma_ddpg': {
        'class': 'MADDPG',
        'policy': 'off-policy',
        'update': 'perStep'
    },
    'ma_td3': {
        'class': 'MATD3',
        'policy': 'off-policy',
        'update': 'perStep'
    }
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
        model_file = importlib.import_module('Algorithms.tf2algos.' + name)
        model = getattr(model_file, algos[name]['class'])
        import os
        import yaml
        filename = f'Algorithms/config.yaml'
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                x = yaml.safe_load(f.read())
        else:
            raise Exception('cannot find this config.')
        algo_config = x[name]
        policy_mode = algos[name]['policy']
        return model, algo_config, policy_mode
