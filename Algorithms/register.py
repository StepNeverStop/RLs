import tensorflow as tf
try:
    tf_version = tf.version.VERSION[0]
except:
    tf_version = tf.VERSION[0]
finally:
    if tf_version == '1':
        from .tf1algos import *
        # algorithms based on TF 1.x
        version = 'tf1algos'
        algos = {
            'pg': [pg_config, PG, 'on-policy', 'perEpisode'],
            'ppo': [ppo_config, PPO, 'on-policy', 'perEpisode'],
            'ac': [ac_config, AC, 'off-policy', 'perStep'],  # could be on-policy, but also doesn't work well.
            'a2c': [a2c_config, A2C, 'on-policy', 'perEpisode'],
            'dpg': [dpg_config, DPG, 'off-policy', 'perStep'],
            'ddpg': [ddpg_config, DDPG, 'off-policy', 'perStep'],
            'td3': [td3_config, TD3, 'off-policy', 'perStep'],
            'sac': [sac_config, SAC, 'off-policy', 'perStep'],
            'sac_no_v': [sac_no_v_config, SAC_NO_V, 'off-policy', 'perStep'],
            'dqn': [dqn_config, DQN, 'off-policy', 'perStep'],
            'ddqn': [ddqn_config, DDQN, 'off-policy', 'perStep'],
            'dddqn': [dddqn_config, DDDQN, 'off-policy', 'perStep'],
            'ma_dpg': [ma_dpg_config, MADPG, 'off-policy', 'perStep'],
            'ma_ddpg': [ma_ddpg_config, MADDPG, 'off-policy', 'perStep'],
            'ma_td3': [ma_td3_config, MATD3, 'off-policy', 'perStep'],
        }
    elif tf_version == '2':
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