import tensorflow as tf

if tf.__version__[0] == '1':
    from .tf1algos import *
    # algorithms based on TF 1.x
    version = 'tf1algos'
    algos = {
        'pg': [PG, 'on-policy', 'perEpisode'],
        'ppo': [PPO, 'on-policy', 'perEpisode'],
        'ac': [AC, 'off-policy', 'perStep'],  # could be on-policy, but also doesn't work well.
        'a2c': [A2C, 'on-policy', 'perEpisode'],
        'dpg': [DPG, 'off-policy', 'perStep'],
        'ddpg': [DDPG, 'off-policy', 'perStep'],
        'td3': [TD3, 'off-policy', 'perStep'],
        'sac': [SAC, 'off-policy', 'perStep'],
        'sac_no_v': [SAC_NO_V, 'off-policy', 'perStep'],
        'dqn': [DQN, 'off-policy', 'perStep'],
        'ddqn': [DDQN, 'off-policy', 'perStep'],
        'dddqn': [DDDQN, 'off-policy', 'perStep'],
        'ma_dpg': [MADPG, 'off-policy', 'perStep'],
        'ma_ddpg': [MADDPG, 'off-policy', 'perStep'],
        'ma_td3': [MATD3, 'off-policy', 'perStep'],
    }
elif tf.__version__[0] == '2':
    from .tf2algos import *
    # algorithms based on TF 2.0
    version = 'tf2algos'
    algos = {
        'pg': [PG, 'on-policy', 'perEpisode'],
        'ppo': [PPO, 'on-policy', 'perEpisode'],
        'ac': [AC, 'off-policy', 'perStep'],  # could be on-policy, but also doesn't work well.
        'a2c': [A2C, 'on-policy', 'perEpisode'],
        'dpg': [DPG, 'off-policy', 'perStep'],
        'ddpg': [DDPG, 'off-policy', 'perStep'],
        'td3': [TD3, 'off-policy', 'perStep'],
        'sac': [SAC, 'off-policy', 'perStep'],
        'sac_no_v': [SAC_NO_V, 'off-policy', 'perStep'],
        'dqn': [DQN, 'off-policy', 'perStep'],
        'ddqn': [DDQN, 'off-policy', 'perStep'],
        'dddqn': [DDDQN, 'off-policy', 'perStep'],
        'maxsqn': [MAXSQN, 'off-policy', 'perStep'],
        'ma_dpg': [MADPG, 'off-policy', 'perStep'],
        'ma_ddpg': [MADDPG, 'off-policy', 'perStep'],
        'ma_td3': [MATD3, 'off-policy', 'perStep'],
    }

def get_model_info(name):
    if name not in algos.keys():
        raise NotImplementedError
    else:
        return version, algos[name]