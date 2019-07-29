# RLs

This project include some state-of-art or classic RL(reinforcement learning) algorithms used for training agents by interactive with Unity through [ml-agents](https://github.com/Unity-Technologies/ml-agents) v0.8.1.

```
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
Example:
    python run.py -a sac -g -e C:/test.exe -p 6666 -s 10 -n test -c config.yaml --max-step 1000
```

For now, those algorithms are available:
- continuous and discrete
- - Policy Gradient, PG, continuous and discrete
- - Actor Critic, AC, continuous and discrete
- - Advantage Actor Critic, A2C, continuous and discrete
- - Proximal Policy Optimization, PPO, continuous and discrete
- continuous
- - Soft Actor Critic, SAC, continuous
- - Twin Delayed Deep Deterministic Policy Gradient, TD3, continuous
- discrete
- Deep Q-learning Network, DQN, discrete
- Deep Deterministic Policy Gradient, DDPG, discrete


Any questions about this project or errors about my bad grammer, plz let me know in [this](https://github.com/StepNeverStop/RLs/issues/new).