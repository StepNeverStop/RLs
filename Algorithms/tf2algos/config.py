dqn_config = {
    'lr': 5.0e-4,
    'gamma': 0.99,
    'epsilon': 0.2,
    'max_episode': 50000,
    'batch_size': 1024,
    'buffer_size': 200000,
    'assign_interval': 1000,
    'use_priority': False,
    'n_step': False
}
ddqn_config = {
    'lr': 5.0e-4,
    'gamma': 0.99,
    'epsilon': 0.2,
    'max_episode': 50000,
    'batch_size': 1024,
    'buffer_size': 200000,
    'assign_interval': 1000,
    'use_priority': False,
    'n_step': False
}
dddqn_config = {
    'lr': 5.0e-4,
    'gamma': 0.99,
    'epsilon': 0.2,
    'max_episode': 50000,
    'batch_size': 1024,
    'buffer_size': 200000,
    'assign_interval': 1000,
    'use_priority': False,
    'n_step': False
}
pg_config = {
    'epsilon': 0.2,
    'lr': 5.0e-4,
    'gamma': 0.99,
    'max_episode': 50000,
    'batch_size': 1024,
    'epoch': 1  # very important
}
ac_config = {
    'lr': 5.0e-4,
    'epsilon': 0.2,
    'gamma': 0.99,
    'max_episode': 50000,
    'batch_size': 1024,
    'buffer_size': 200000,
    'use_priority': False,
    'n_step': False
}
a2c_config = {
    'epsilon': 0.2,
    'lr': 5.0e-4,
    'gamma': 0.99,
    'beta': 1.0e-3,
    'max_episode': 50000,
    'batch_size': 1024,
    'epoch': 1  # very important
}
ppo_config = {
    'epsilon': 0.2,
    'gamma': 0.99,
    'beta': 1.0e-3,
    'lr': 5.0e-4,
    'lambda_': 0.95,
    'max_episode': 50000,
    'batch_size': 1024,
    'share_net': True,
    'epoch': 4  # very important
}
dpg_config = {
    'gamma': 0.99,
    'lr': 5.0e-4,
    'max_episode': 50000,
    'batch_size': 1024,
    'buffer_size': 200000,
    'use_priority': False,
    'n_step': False
}
ddpg_config = {
    'gamma': 0.99,
    'ployak': 0.995,
    'lr': 5.0e-4,
    'max_episode': 50000,
    'batch_size': 1024,
    'buffer_size': 200000,
    'use_priority': False,
    'n_step': False
}
td3_config = {
    'gamma': 0.99,
    'ployak': 0.995,
    'lr': 5.0e-4,
    'max_episode': 50000,
    'batch_size': 1024,
    'buffer_size': 200000,
    'use_priority': False,
    'n_step': False
}
sac_config = {
    'alpha': 0.2,
    'auto_adaption': True,
    'gamma': 0.99,
    'ployak': 0.995,
    'lr': 5.0e-4,
    'max_episode': 50000,
    'batch_size': 1024,
    'buffer_size': 200000,
    'use_priority': False,
    'n_step': False
}
sac_no_v_config = {
    'alpha': 0.2,
    'auto_adaption': True,
    'lr': 5.0e-4,
    'max_episode': 50000,
    'gamma': 0.99,
    'ployak': 0.995,
    'batch_size': 1024,
    'buffer_size': 200000,
    'use_priority': False,
    'n_step': False
}
maxsqn_config = {
    'alpha': 0.2,
    'epsilon': 0.2,
    'use_epsilon': False,
    'auto_adaption': True,
    'lr': 5.0e-4,
    'max_episode': 50000,
    'gamma': 0.99,
    'ployak': 0.995,
    'batch_size': 1024,
    'buffer_size': 200000,
    'use_priority': False,
    'n_step': False
}
ma_dpg_config = {
    'gamma': 0.99,
    'lr': 5.0e-4,
    'max_episode': 50000
}
ma_ddpg_config = {
    'gamma': 0.99,
    'ployak': 0.995,
    'lr': 5.0e-4,
    'max_episode': 50000
}
ma_td3_config = {
    'gamma': 0.99,
    'ployak': 0.995,
    'lr': 5.0e-4,
    'max_episode': 50000
}
