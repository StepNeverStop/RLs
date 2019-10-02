# RLs

:evergreen_tree::evergreen_tree::evergreen_tree:This project include some state-of-the-art or classic RL(reinforcement learning) algorithms used for training agents by interacting with Unity through [ml-agents](https://github.com/Unity-Technologies/ml-agents/tree/0.10.0) v0.10.0 or with [gym](https://github.com/openai/gym). The goal of this framework is to provide stable implementations of standard RL algorithms and simultaneously enable fast prototyping of new methods.

## About

It aims to fill the need for a small, easily grokked codebase in which users can freely experiment with wild ideas (speculative research).

### Characteristics

- Suitable for Windows, Linux, and OSX
- **Support TensorFlow 1.x and TensorFlow 2.0**
- Almost reimplementation and competitive performance of original papers
- Reusable modules
- Clear hierarchical structure and easy code control
- Compatible with OpenAI Gym, Unity3D Ml-agents
- Restoring the training process from where it stopped, retraining on a new task, fine-tuning
- Using other training tsak's model as parameter initialization `--load`

### Supports

This project supports:
- Unity3D ml-agents.
- Gym, for now only two data types are compatible——`[Box, Discrete]`. Support 99.65% environment settings of Gym(except `Blackjack-v0`, `KellyCoinflip-v0`, and `KellyCoinflipGeneralized-v0`). **Support parallel training using gym envs, just need to specify `--gym-agents` to how many agents you want to train in parallel.**
    - Discrete -> Discrete
    - Discrete -> Box
    - Box -> Discrete
    - Box -> Box
    - Box/Discrete -> Tuple(Discrete, Discrete, Discrete)
- MultiAgent training. One brain controls multiple agents.
- MultiBrain training. Brains' model should be same algorithm or have the same learning-progress(perStep or perEpisode).
- MultiImage input. Images should have the same input format, like `[84, 84, 3]` (only for ml-agents).
- Four types of ReplayBuffer, Default is ER, using other RBs need to modify the code a little: 
    - ER
    - n-step ER
    - [Prioritized ER](https://arxiv.org/abs/1511.05952)
    - n-step Prioritized ER

### Advantages

- Parallel training multiple scenes for Gym
- Unified data format of environments between ml-agents and gym
- Just need to write a single file for other algorithms' implementation(Similar algorithm structure).
- Many controllable factors and adjustable parameters

## Getting started

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
    --load=<name>               指定载入model的训练名称 [default: None]
    --gym                       是否使用gym训练环境 [default: False]
    --gym-agents=<n>            指定并行训练的数量 [default: 1]
    --gym-env=<name>            指定gym环境的名字 [default: CartPole-v0]
    --render-episode=<n>        指定gym环境从何时开始渲染 [default: None]
Example:
    python run.py -a sac -g -e C:/test.exe -p 6666 -s 10 -n test -c config.yaml --max-step 1000 --sampler C:/test_sampler.yaml
    python run.py -a ppo -u -n train_in_unity --load last_train_name
    python run.py -ui -a td3 -n inference_in_unity
    python run.py -gi -a dddqn -n inference_with_build -e my_executable_file.exe
    python run.py --gym -a ppo -n train_using_gym --gym-env MountainCar-v0 --render-episode 1000 --gym-agents 4
"""
```

If you specify gym, unity, and env simultaneously, the following priorities will be followed: gym > unity > unity_env.

## Implemented Algorithms

For now, those algorithms are available:
- CONTINUOUS AND DISCRETE
    - :bug:Policy Gradient, PG
    - :bug:Actor Critic, AC
    - Advantage Actor Critic, A2C
    - :boom:[Proximal Policy Optimization, PPO](https://arxiv.org/abs/1707.06347)
- CONTINUOUS
    - [Deterministic Policy Gradient, DPG](https://hal.inria.fr/file/index/docid/938992/filename/dpg-icml2014.pdf)
    - [Deep Deterministic Policy Gradient, DDPG](https://arxiv.org/abs/1509.02971)
    - :fire:[Soft Actor Critic, SAC](https://arxiv.org/abs/1812.05905)
    - :fire:[Twin Delayed Deep Deterministic Policy Gradient, TD3](https://arxiv.org/abs/1802.09477)
- DISCRETE
    - Deep Q-learning Network, DQN, [2013](https://arxiv.org/pdf/1312.5602.pdf), [2015](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    - [Double Deep Q-learning Network, DDQN](https://arxiv.org/abs/1509.06461)
    - [Dueling Double Deep Q-learning Network, DDDQN](https://arxiv.org/abs/1511.06581)

Multi-Agent training algorithms(*not support visual input yet*):
- [Multi-Agent Deep Deterministic Policy Gradient, MADDPG](https://arxiv.org/abs/1706.02275)
- Multi-Agent Deterministic Policy Gradient, MADPG
- Multi-Agent Twin Delayed Deep Deterministic Policy Gradient, MATD3

## Notes

1. log, model, training parameter configuration, and data are stored in `C:/RLdata` for Windows, or `/RLdata` for Linux/OSX
2. need to use command `su` or `sudo` to run on a Linux/OSX
3. record directory format is `RLdata/Environment/Algorithm/Brain name(for ml-agents)/Training name/config&excel&log&model`
4. make sure brains' number > 1 if specifing `ma*` algorithms like maddpg
5. maddpg doesn't support visual input for now
6. need 3 steps to implement a new algorithm
    1. write `.py` in `Algorithms/tf[x]algos` directory and make the policy inherit from class `Base` or `Policy`, add `from .[name] import [name]` in `Algorithms/tf[x]algos/__init__.py`
    2. write default configuration in `Algorithms/config.py`
    3. register new algorithm in `algos` of `run.py`
7. set algorithms' hyper-parameters in [Algorithms/config.py](https://github.com/StepNeverStop/RLs/blob/master/Algorithms/config.py)
8. set training default configuration in [config.py](https://github.com/StepNeverStop/RLs/blob/master/config.py)
9. change neural network structure in [Nn/nn.py](https://github.com/StepNeverStop/RLs/blob/master/Nn/nn.py)

## Ongoing things

- RNN
- Better episode experience replay buffer
- TRPO

## Installation

### Dependencies

- python>3.6, <3.7
- tensorflow>=1.7.0, <=1.12.0 or tensorflow==2.0.0-rc1
- pandas
- numpy
- pywin32==224
- docopt
- pyyaml
- pillow
- openpyxl

### Install

```bash
$ git clone https://github.com/StepNeverStop/RLs.git
```
pip package coming soon.

## Giving credit

If you use this repository for you research, please cite:
```
@misc{RLs,
  author = {Keavnn},
  title = {RLs: Reinforcement Learning search framework for Unity3D and Gym},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/StepNeverStop/RLs}},
}
```

## Issues

Any questions about this project or errors about my bad grammer, plz let me know in [this](https://github.com/StepNeverStop/RLs/issues/new).
