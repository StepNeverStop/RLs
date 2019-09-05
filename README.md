# RLs

:evergreen_tree::evergreen_tree::evergreen_tree:This project include some state-of-art or classic RL(reinforcement learning) algorithms used for training agents by interactive with Unity through [ml-agents](https://github.com/Unity-Technologies/ml-agents/tree/0.9.0) v0.9.0 or through gym.

This project support:
- Unity3D ml-agents.
- Gym, for now only two data types are compatible——`[Box, Discrete]`.
- MultiAgent training. One brain controls multiple agents.
- MultiBrain training. Brains' model should be same algorithm or have the same learning-progress(perStep or perEpisode).
- MultiImage input. Images should have the same input format, like `[84, 84, 3]`.
- Four types of ReplayBuffer: ER, n-step ER, PER, n-step PER. Default is ER, using other RBs need to modify the code a little.

```python
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
    --gym                       是否使用gym训练环境 [default: False]
    --gym-env=<name>            指定gym环境的名字 [default: CartPole-v0]
Example:
    python run.py -a sac -g -e C:/test.exe -p 6666 -s 10 -n test -c config.yaml --max-step 1000 --sampler C:/test_sampler.yaml
    python run.py -a ppo -u -n train_in_unity
    python run.py -ui -a td3 -n inference_in_unity
    python run.py -gi -a dddqn -n inference_with_build -e my_executable_file.exe
    python run.py --gym -a ddpg -n train_using_gym --gym-env=MountainCar-v0
"""
```

If you specify gym, unity, and env, the following priorities will be followed: gym > unity > unity_env.

For now, those algorithms are available:
- CONTINUOUS AND DISCRETE
- - :bug:Policy Gradient, PG
- - :bug:Actor Critic, AC
- - Advantage Actor Critic, A2C
- - :boom:Proximal Policy Optimization, PPO
- CONTINUOUS
- - Deterministic Policy Gradient, DPG
- - Deep Deterministic Policy Gradient, DDPG
- - :fire:Soft Actor Critic, SAC​​
- - :fire:Twin Delayed Deep Deterministic Policy Gradient, TD3
- DISCRETE
- - Deep Q-learning Network, DQN
- - Double Deep Q-learning Network, DDQN
- - Dueling Double Deep Q-learning Network, DDDQN

Any questions about this project or errors about my bad grammer, plz let me know in [this](https://github.com/StepNeverStop/RLs/issues/new).