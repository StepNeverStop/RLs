# RLs

:evergreen_tree::evergreen_tree::evergreen_tree:This project include some state-of-art or classic RL(reinforcement learning) algorithms used for training agents by interactive with Unity through [ml-agents](https://github.com/Unity-Technologies/ml-agents/tree/0.9.0) v0.9.0.

"""
Usage:
    python [options]

Options:
    -h,--help                   显示帮助
    -i,--inference              推断 [default: False]
    -a,--algorithm=<name>       算法 [default: ppo]
    -c,--config-file=<file>     指定模型的超参数config文件 [default: None]
    -e,--env=<file>             指定环境名称 [default: None]
    -p,--port=<n>               端口 [default: 5005]
    -u,--unity                  是否使用unity客户端 [default: False]
    -g,--graphic                是否显示图形界面 [default: False]
    -n,--name=<name>            训练的名字 [default: None]
    -s,--save-frequency=<n>     保存频率 [default: None]
    --max-step=<n>              每回合最大步长 [default: None]
    --sampler=<file>            指定随机采样器的文件路径 [default: None]
Example:
    python run.py -a sac -g -e C:/test.exe -p 6666 -s 10 -n test -c config.yaml --max-step 1000 --sampler C:/test_sampler.yaml
"""

For now, those algorithms are available:
- CONTINUOUS AND DISCRETE
- - :bug:Policy Gradient, PG
- - :bug:Actor Critic, AC
- - Advantage Actor Critic, A2C
- - :boom:Proximal Policy Optimization, PPO
- CONTINUOUS
- - :fire:Soft Actor Critic, SAC​​
- - :fire:Twin Delayed Deep Deterministic Policy Gradient, TD3
- DISCRETE
- - Deep Q-learning Network, DQN
- - Double Deep Q-learning Network, DDQN
- - Dueling Double Deep Q-learning Network, DDDQN
- - Deep Deterministic Policy Gradient, DDPG


Any questions about this project or errors about my bad grammer, plz let me know in [this](https://github.com/StepNeverStop/RLs/issues/new).