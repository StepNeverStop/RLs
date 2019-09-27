dqn_config = {
    'lr': 5.0e-4,
    'gamma': 0.99,
    'epsilon': 0.2,
    'max_episode': 50000,
    'batch_size': 100,
    'buffer_size': 10000,
    'assign_interval': 1000,
    'use_priority': False,
    'n_step': False
}
ddqn_config = {
    'lr': 5.0e-4,
    'gamma': 0.99,
    'epsilon': 0.2,
    'max_episode': 50000,
    'batch_size': 100,
    'buffer_size': 10000,
    'assign_interval': 1000
}
dddqn_config = {
    'lr': 5.0e-4,
    'gamma': 0.99,
    'epsilon': 0.2,
    'max_episode': 50000,
    'batch_size': 100,
    'buffer_size': 10000,
    'assign_interval': 1000
}
pg_config = {
    'lr': 5.0e-4,
    'gamma': 0.99,
    'max_episode': 50000,
    'batch_size': 500,
    'epoch': 1
}
ac_config = {
    'lr': 5.0e-4,
    'gamma': 0.99,
    'max_episode': 50000,
    'batch_size': 100,
    'buffer_size': 10000,
}
a2c_config = {
    'lr': 5.0e-4,
    'gamma': 0.99,
    'max_episode': 50000,
    'batch_size': 500,
}
ppo_config = {
    'epsilon': 0.2,
    'gamma': 0.99,
    'beta': 1.0e-3,
    'lr': 5.0e-4,
    'lambda_': 0.95,
    'max_episode': 50000,
    'batch_size': 100,
    'epoch': 5
}
dpg_config = {
    'gamma': 0.99,
    'lr': 5.0e-4,
    'max_episode': 50000,
    'batch_size': 100,
    'buffer_size': 10000
}
ddpg_config = {
    'gamma': 0.99,
    'ployak': 0.995,
    'lr': 5.0e-4,
    'max_episode': 50000,
    'batch_size': 100,
    'buffer_size': 10000
}
td3_config = {
    'gamma': 0.99,
    'ployak': 0.995,
    'lr': 5.0e-4,
    'max_episode': 50000,
    'batch_size': 100,
    'buffer_size': 10000,
}
sac_config = {
    'alpha': 0.2,
    'auto_adaption': True,
    'gamma': 0.99,
    'ployak': 0.995,
    'lr': 5.0e-4,
    'max_episode': 50000,
    'batch_size': 100,
    'buffer_size': 10000
}
sac_no_v_config = {
    'alpha': 0.2,
    'auto_adaption': True,
    'lr': 5.0e-4,
    'max_episode': 50000,
    'gamma': 0.99,
    'ployak': 0.995,
    'batch_size': 100,
    'buffer_size': 10000
}
madpg_config = {
    'gamma': 0.99,
    'lr': 5.0e-4,
    'max_episode': 50000
}
maddpg_config = {
    'gamma': 0.99,
    'ployak': 0.995,
    'lr': 5.0e-4,
    'max_episode': 50000
}
matd3_config = {
    'gamma': 0.99,
    'ployak': 0.995,
    'lr': 5.0e-4,
    'max_episode': 50000
}