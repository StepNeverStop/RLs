<div align="center">
	<a href="https://github.com/StepNeverStop/RLs">
		<img width="auto" height="200px" src="./pics/logo.png">
	</a>
	<br/>
	<br/>
	<a href="https://github.com/StepNeverStop/RLs">
		<img width="auto" height="20px" src="./pics/font.png">
	</a>
</div>

<div align="center">
<p><strong>RLs:</strong> Reinforcement Learning Algorithm Based On PyTorch.</p> 
</div>

# RLs

This project includes SOTA or classic reinforcement learning (single and multi-agent) algorithms used for training
agents by interacting with Unity through [ml-agents](https://github.com/Unity-Technologies/ml-agents/tree/release_18)
Release 18 or with [gym](https://github.com/openai/gym).

## About

The goal of this framework is to provide stable implementations of standard RL algorithms and simultaneously enable fast
prototyping of new methods. It aims to fill the need for a small, easily grokked codebase in which users can freely
experiment with wild ideas (speculative research).

## Characteristics

This project supports:

- Suitable for Windows, Linux, and OSX
- Single- and Multi-Agent training.
- Multiple type of observation sensors as input.
- Only need 3 steps to implement a new algorithm:
    1. **policy** write `.py` in `rls/algorithms/{single/multi}` directory and make the policy inherit from super-class
       defined in `rls/algorithms/base`
    2. **config** write default configuration in `rls/configs/algorithms.yaml`
    3. **register** register new algorithm in `rls/algorithms/__init__.py`
- Only need 3 steps to adapt to a new training environment:
    1. **wrapper** write environment wrappers in `rls/envs/{new platform}` directory and make it inherit from
       super-class defined in `rls/envs/env_base.py`
    2. **config** write default configuration in `rls/configs/{new platform}`
    3. **register** register new environment platform in `rls/envs/__init__.py`
- Compatible with several environment platforms
    - [Unity3D ml-agents](https://github.com/Unity-Technologies/ml-agents).
    - [PettingZoo](https://www.pettingzoo.ml/#)
    - [gym](https://github.com/openai/gym), for now only two data types are compatible——`[Box, Discrete]`. Support
      parallel training using gym envs, just need to specify `--copies` to how many agents you want to train in
      parallel.
        - environments:
            - [MuJoCo](https://github.com/openai/mujoco-py)(v2.0.2.13)
            - [PyBullet](https://github.com/bulletphysics/bullet3)
            - [gym_minigrid](https://github.com/maximecb/gym-minigrid)
        - observation -> action:
            - Discrete -> Discrete (observation type -> action type)
            - Discrete -> Box
            - Box -> Discrete
            - Box -> Box
            - Box/Discrete -> Tuple(Discrete, Discrete, Discrete)
- Four types of Replay Buffer, Default is ER:
    - ER
    - [Prioritized ER](https://arxiv.org/abs/1511.05952)
- [Noisy Net](https://arxiv.org/abs/1706.10295) for better exploration.
- [Intrinsic Curiosity Module](https://arxiv.org/abs/1705.05363) for almost all off-policy algorithms implemented.
- Parallel training multiple scenes for Gym
- Unified data format

## Installation

method 1:

```bash
$ git clone https://github.com/StepNeverStop/RLs.git
$ cd RLs
$ conda create -n rls python=3.8
$ conda activate rls
# Windows
$ pip install -e .[windows]
# Linux or Mac OS
$ pip install -e .
```

method 1:

```bash
conda env create -f environment.yaml
```

If using ml-agents:

```bash
$ pip install -e .[unity]
```

You can download the builded docker image from [here](https://hub.docker.com/r/keavnn/rls):

```bash
$ docker pull keavnn/rls:latest
```

If anyone who wants to send a PR, plz format all code-files first:

```bash
$ pip install -e .[pr]
$ python auto_format.py -d ./
```

## Implemented Algorithms

For now, these algorithms are available:

- Multi-Agent training algorithms:
    - Independent-SARL, i.e. IQL, [I-DQN](http://arxiv.org/abs/1511.08779), etc.
    - [Value-Decomposition Networks, VDN](http://arxiv.org/abs/1706.05296)
    - [Monotonic Value Function Factorisation Networks, QMIX](http://arxiv.org/abs/1803.11485)
    - [Multi-head Attention based Q-value Mixing Network, Qatten](http://arxiv.org/abs/2002.03939)
    - [Factorize with Transformation, Qtran](https://arxiv.org/abs/1905.05408)
    - [Duplex Dueling Multi-Agent Q-Learning, QPLEX](http://arxiv.org/abs/2008.01062)
    - [Multi-Agent Deep Deterministic Policy Gradient, MADDPG](https://arxiv.org/abs/1706.02275)
- Single-Agent training algorithms(Some algorithms that only support continuous space problems use Gumbel-softmax trick
  to implement discrete versions, i.e. DDPG):
    - Policy Gradient, PG
    - Actor Critic, AC
    - [Synchronous Advantage Actor Critic, A2C](http://arxiv.org/abs/1602.01783)
    <!-- - [Trust Region Policy Optimization, TRPO](https://arxiv.org/abs/1502.05477) -->
    - :boom:Proximal Policy Optimization, [PPO](https://arxiv.org/abs/1707.06347)
      , [DPPO](http://arxiv.org/abs/1707.02286,)
    - [Trust Region Policy Optimization, TRPO](https://arxiv.org/abs/1502.05477)
    - [Natural Policy Gradient, NPG](https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf)
    - [Deterministic Policy Gradient, DPG](https://hal.inria.fr/file/index/docid/938992/filename/dpg-icml2014.pdf)
    - [Deep Deterministic Policy Gradient, DDPG](https://arxiv.org/abs/1509.02971)
    - :fire:Soft Actor Critic, [SAC](https://arxiv.org/abs/1812.05905), [Discrete SAC](https://arxiv.org/abs/1910.07207)
    - [Tsallis Actor Critic, TAC](https://arxiv.org/abs/1902.00137)
    - :fire:[Twin Delayed Deep Deterministic Policy Gradient, TD3](https://arxiv.org/abs/1802.09477)
    - Deep Q-learning Network, DQN, [2013](https://arxiv.org/pdf/1312.5602.pdf)
      , [2015](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    - [Double Deep Q-learning Network, DDQN](https://arxiv.org/abs/1509.06461)
    - [Dueling Double Deep Q-learning Network, DDDQN](https://arxiv.org/abs/1511.06581)
    - [Deep Recurrent Q-learning Network, DRQN](https://arxiv.org/abs/1507.06527)
    - [Deep Recurrent Double Q-learning, DRDQN](https://arxiv.org/abs/1908.06040)
    - [Category 51, C51](https://arxiv.org/abs/1707.06887)
    - [Quantile Regression DQN, QR-DQN](https://arxiv.org/abs/1710.10044)
    - [Implicit Quantile Networks, IQN](https://arxiv.org/abs/1806.06923)
    - [Rainbow DQN](https://arxiv.org/abs/1710.02298)
    - [MaxSQN](https://github.com/createamind/DRL/blob/master/spinup/algos/maxsqn/maxsqn.py)
    - [Soft Q-Learning, SQL](https://arxiv.org/abs/1702.08165)
    - [Bootstrapped DQN](http://arxiv.org/abs/1602.04621)
    - [Averaged DQN](http://arxiv.org/abs/1611.01929)
    - Hierachical training algorithms:
        - [Option-Critic, OC](http://arxiv.org/abs/1609.05140)
        - [Asynchronous Advantage Option-Critic, A2OC](http://arxiv.org/abs/1709.04571)
        - [PPO Option-Critic, PPOC](http://arxiv.org/abs/1712.00004)
        - [Interest-Option-Critic, IOC](http://arxiv.org/abs/2001.00271)
    - Model-based algorithms:
        - [Learning Latent Dynamics for Planning from Pixels, PlaNet](http://arxiv.org/abs/1811.04551)
        - [Dream to Control, Dreamer](http://arxiv.org/abs/1912.01603)
        - [Mastering Atari with Discrete World Models, DreamerV2](http://arxiv.org/abs/2010.02193)
        - [Model-Based Value Estimation, MVE](http://arxiv.org/abs/1803.00101)
    - Offline algorithms(**under implementation**):
        - [Conservative Q-Learning for Offline Reinforcement Learning, CQL](http://arxiv.org/abs/2006.04779)
        - BCQ
            - Benchmarking Batch Deep Reinforcement Learning Algorithms, [Discrete](http://arxiv.org/abs/1910.01708)
            - Off-Policy Deep Reinforcement Learning without Exploration, [Continuous](http://arxiv.org/abs/1812.02900)

| Algorithms | Discrete | Continuous | Image | RNN | Command parameter | | :-----------------------------: | :------:
| :--------: | :---: | :--: | :---------------: | | PG | ✓ | ✓ | ✓ | ✓ | pg | | AC | ✓ | ✓ | ✓ | ✓ | ac | | A2C | ✓ | ✓
| ✓ | ✓ | a2c | | NPG | ✓ | ✓ | ✓ | ✓ | npg | | TRPO | ✓ | ✓ | ✓ | ✓ | trpo | | PPO | ✓ | ✓ | ✓ | ✓ | ppo | | DQN | ✓ |
| ✓ | ✓ | dqn | | Double DQN | ✓ | | ✓ | ✓ | ddqn | | Dueling Double DQN | ✓ | | ✓ | ✓ | dddqn | | Averaged DQN | ✓ | |
✓ | ✓ | averaged_dqn | | Bootstrapped DQN | ✓ | | ✓ | ✓ | bootstrappeddqn | | Soft Q-Learning | ✓ | | ✓ | ✓ | sql | |
C51 | ✓ | | ✓ | ✓ | c51 | | QR-DQN | ✓ | | ✓ | ✓ | qrdqn | | IQN | ✓ | | ✓ | ✓ | iqn | | Rainbow | ✓ | | ✓ | ✓ | rainbow
| | DPG | ✓ | ✓ | ✓ | ✓ | dpg | | DDPG | ✓ | ✓ | ✓ | ✓ | ddpg | | TD3 | ✓ | ✓ | ✓ | ✓ | td3 | | SAC(has V network)
| ✓ | ✓ | ✓ | ✓ | sac_v | | SAC | ✓ | ✓ | ✓ | ✓ | sac | | TAC | sac | ✓ | ✓ | ✓ | tac | | MaxSQN | ✓ | | ✓ | ✓ | maxsqn
| | OC | ✓ | ✓ | ✓ | ✓ | oc | | AOC | ✓ | ✓ | ✓ | ✓ | aoc | | PPOC | ✓ | ✓ | ✓ | ✓ | ppoc | | IOC | ✓ | ✓ | ✓ | ✓ | ioc
| | PlaNet | ✓ | | ✓ | 1 | planet | | Dreamer | ✓ | ✓ | ✓ | 1 | dreamer | | DreamerV2 | ✓ | ✓ | ✓ | 1 | dreamerv2 | |
MVE | ✓ | ✓ | | | mve | | VDN | ✓ | | ✓ | ✓ | vdn | | QMIX | ✓ | | ✓ | ✓ | qmix | | Qatten | ✓ | | ✓ | ✓ | qatten | |
QPLEX | ✓ | | ✓ | ✓ | qplex | | QTRAN | ✓ | | ✓ | ✓ | qtran | | MADDPG | ✓ | ✓ | ✓ | ✓ | maddpg | | MASAC | ✓ | ✓ | ✓ |
✓ | masac | | CQL | ✓ | | ✓ | ✓ | cql_dqn | | BCQ | ✓ | ✓ | ✓ | ✓ | bcq |

*1 means must use rnn or rnn is used by default.*

## Getting started

```python
"""
usage: run.py [-h] [-c COPIES] [--seed SEED] [-r]
              [-p {gym,unity,pettingzoo}]
              [-a {maddpg,masac,vdn,qmix,qatten,qtran,qplex,aoc,ppoc,oc,ioc,planet,dreamer,dreamerv2,mve,cql_dqn,bcq,pg,npg,trpo,ppo,a2c,ac,dpg,ddpg,td3,sac_v,sac,tac,dqn,ddqn,dddqn,averaged_dqn,c51,qrdqn,rainbow,iqn,maxsqn,sql,bootstrappeddqn}]
              [-i] [-l LOAD_PATH] [-m MODELS] [-n NAME]
              [--config-file CONFIG_FILE] [--store-dir STORE_DIR]
              [--episode-length EPISODE_LENGTH] [--hostname] [-e ENV_NAME]
              [-f FILE_NAME] [-s] [-d DEVICE] [-t MAX_TRAIN_STEP]

optional arguments:
  -h, --help            show this help message and exit
  -c COPIES, --copies COPIES
                        nums of environment copies that collect data in
                        parallel
  --seed SEED           specify the random seed of module random, numpy and
                        pytorch
  -r, --render          whether render game interface
  -p {gym,unity,pettingzoo}, --platform {gym,unity,pettingzoo}
                        specify the platform of training environment
  -a {maddpg,masac,vdn,qmix,qatten,qtran,qplex,aoc,ppoc,oc,ioc,planet,dreamer,dreamerv2,mve,cql_dqn,bcq,pg,npg,trpo,ppo,a2c,ac,dpg,ddpg,td3,sac_v,sac,tac,dqn,ddqn,dddqn,averaged_dqn,c51,qrdqn,rainbow,iqn,maxsqn,sql,bootstrappeddqn}, --algorithm {maddpg,masac,vdn,qmix,qatten,qtran,qplex,aoc,ppoc,oc,ioc,planet,dreamer,dreamerv2,mve,cql_dqn,bcq,pg,npg,trpo,ppo,a2c,ac,dpg,ddpg,td3,sac_v,sac,tac,dqn,ddqn,dddqn,averaged_dqn,c51,qrdqn,rainbow,iqn,maxsqn,sql,bootstrappeddqn}
                        specify the training algorithm
  -i, --inference       inference the trained model, not train policies
  -l LOAD_PATH, --load-path LOAD_PATH
                        specify the name of pre-trained model that need to
                        load
  -m MODELS, --models MODELS
                        specify the number of trails that using different
                        random seeds
  -n NAME, --name NAME  specify the name of this training task
  --config-file CONFIG_FILE
                        specify the path of training configuration file
  --store-dir STORE_DIR
                        specify the directory that store model, log and
                        others
  --episode-length EPISODE_LENGTH
                        specify the maximum step per episode
  --hostname            whether concatenate hostname with the training name
  -e ENV_NAME, --env-name ENV_NAME
                        specify the environment name
  -f FILE_NAME, --file-name FILE_NAME
                        specify the path of builded training environment of
                        UNITY3D
  -s, --save            specify whether save models/logs/summaries while
                        training or not
  -d DEVICE, --device DEVICE
                        specify the device that operate Torch.Tensor
  -t MAX_TRAIN_STEP, --max-train-step MAX_TRAIN_STEP
                        specify the maximum training steps
"""
```

Example:

```bash
python run.py -s    # save model and log while train
python run.py -p gym -a dqn -e CartPole-v0 -c 12 -n dqn_cartpole
python run.py -p unity -a ppo -n run_with_unity -c 1
```

The main training loop of **pseudo-code** in this repo is as:

```python
# noinspection PyUnresolvedReferences
agent.episode_reset()  # initialize rnn hidden state or something else
# noinspection PyUnresolvedReferences
obs = env.reset()
while True:
    # noinspection PyUnresolvedReferences
    env_rets = env.step(agent(obs))
    # noinspection PyUnresolvedReferences
    agent.episode_step(obs, env_rets)  # store experience, save model, and train off-policy algorithms
    obs = env_rets['obs']
    if env_rets['done']:
        break
# noinspection PyUnresolvedReferences
agent.episode_end()  # train on-policy algorithms
```

## Giving credit

If using this repository for your research, please cite:

```
@misc{RLs,
  author = {Keavnn},
  title = {RLs: A Featureless Reinforcement Learning Repository},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/StepNeverStop/RLs}},
}
```

## Issues

Any questions/errors about this project, please let me know in [here](https://github.com/StepNeverStop/RLs/issues/new).
