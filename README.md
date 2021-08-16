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

NOTE: This repo in under reconstruction to better compat with Multi-Agent reinforcement learning. Bugs will not be fixed on time.

This project includes SOTA or classic RL(reinforcement learning) algorithms used for training agents by interacting with Unity through [ml-agents](https://github.com/Unity-Technologies/ml-agents/tree/release_18) Release 18 or with [gym](https://github.com/openai/gym). The goal of this framework is to provide stable implementations of standard RL algorithms and simultaneously enable fast prototyping of new methods.

![](./pics/framework.jpg)

## About

It aims to fill the need for a small, easily grokked codebase in which users can freely experiment with wild ideas (speculative research).

### Characteristics

- Suitable for Windows, Linux, and OSX
- Almost reimplementation and competitive performance of original papers
- Reusable modules
- Clear hierarchical structure and easy code control
- Compatible with OpenAI Gym and Unity3D Ml-agents
- Restoring the training process from where it stopped, retraining on a new task, fine-tuning
- Using other training task's model as parameter initialization, specifying `--load`

### Supports

This project supports:
- Unity3D ml-agents.
- Gym{MuJoCo(v2.0.2.13), [PyBullet](https://github.com/bulletphysics/bullet3), [gym_minigrid](https://github.com/maximecb/gym-minigrid)}, for now only two data types are compatible——`[Box, Discrete]`. Support parallel training using gym envs, just need to specify `--copys` to how many agents you want to train in parallel.
    - Discrete -> Discrete (observation type -> action type)
    - Discrete -> Box
    - Box -> Discrete
    - Box -> Box
    - Box/Discrete -> Tuple(Discrete, Discrete, Discrete)
- MultiAgent training.
- MultiImage input. Images will resized to same shape before store into replay buffer, like `[84, 84, 3]`.
- Four types of Replay Buffer, Default is ER: 
    - ER
    - n-step ER
    - [Prioritized ER](https://arxiv.org/abs/1511.05952)
    - n-step Prioritized ER
- [Noisy Net](https://arxiv.org/abs/1706.10295) for better exploration.
- [Intrinsic Curiosity Module](https://arxiv.org/abs/1705.05363) for almost all off-policy algorithms implemented.

### Advantages

- Parallel training multiple scenes for Gym
- Unified data format of environments between ml-agents and gym
- Just need to write a single file for other algorithms' implementation(Similar algorithm structure).
- Many controllable factors and adjustable parameters

## Installation

method 1:
```bash
conda env create -f environment.yaml
```

method 2:
```bash
$ git clone https://github.com/StepNeverStop/RLs.git
$ cd RLs
$ conda create -n rls python=3.6
$ conda activate rls
# Windows
$ pip install -e .[windows]
# Linux or Mac OS
$ pip install -e .
```

If using ml-agents:
```bash
$ pip install -e .[unity]
```

If using mujoco:
```bash
$ pip install -e .[mujoco]
```

If using atari/box2d:
```bash
$ pip install gym[atari]
$ pip install gym[box2d]
```

You can download the builded docker image from [here](https://hub.docker.com/r/keavnn/rls):
```bash
$ docker pull keavnn/rls:latest
```

## Implemented Algorithms

For now, these algorithms are available:

- Single-Agent training algorithms(Some algorithms that only support continuous space problems use Gumbel-softmax trick to implement discrete versions, i.e. DDPG):
    - Q-Learning, Sarsa, Expected Sarsa
    - :bug:Policy Gradient, PG
    - :bug:Actor Critic, AC
    - Advantage Actor Critic, A2C
    - [Trust Region Policy Optimization, TRPO](https://arxiv.org/abs/1502.05477)
    - :boom:Proximal Policy Optimization, [PPO](https://arxiv.org/abs/1707.06347), [DPPO](http://arxiv.org/abs/1707.02286,)
    - [Deterministic Policy Gradient, DPG](https://hal.inria.fr/file/index/docid/938992/filename/dpg-icml2014.pdf)
    - [Deep Deterministic Policy Gradient, DDPG](https://arxiv.org/abs/1509.02971)
    - :fire:Soft Actor Critic, [SAC](https://arxiv.org/abs/1812.05905), [Discrete SAC](https://arxiv.org/abs/1910.07207)
    - [Tsallis Actor Critic, TAC](https://arxiv.org/abs/1902.00137)
    - :fire:[Twin Delayed Deep Deterministic Policy Gradient, TD3](https://arxiv.org/abs/1802.09477)
    - Deep Q-learning Network, DQN, [2013](https://arxiv.org/pdf/1312.5602.pdf), [2015](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
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
    - [Contrastive Unsupervised RL, CURL](http://arxiv.org/abs/2004.04136)
- Hierachical training algorithms:
    - [Option-Critic, OC](http://arxiv.org/abs/1609.05140)
    - [Asynchronous Advantage Option-Critic, A2OC](http://arxiv.org/abs/1709.04571)
    - [PPO Option-Critic, PPOC](http://arxiv.org/abs/1712.00004)
    - [Interest-Option-Critic, IOC](http://arxiv.org/abs/2001.00271)
    - [HIerarchical Reinforcement learning with Off-policy correction, HIRO](http://arxiv.org/abs/1805.08296)
- Multi-Agent training algorithms:
    - [Multi-Agent Deep Deterministic Policy Gradient, MADDPG](https://arxiv.org/abs/1706.02275)



|         Algorithms(30)          | Discrete | Continuous | Image | RNN  | Command parameter |
| :-----------------------------: | :------: | :--------: | :---: | :--: | :---------------: |
| Q-Learning/Sarsa/Expected Sarsa |    √     |            |       |      |        qs         |
|            ~~CEM~~              |    √     |     √      |       |      |        cem        |
|               PG                |    √     |     √      |   √   |      |        pg         |
|               AC                |    √     |     √      |   √   |  √   |        ac         |
|               A2C               |    √     |     √      |   √   |      |        a2c        |
|              TRPO               |    √     |     √      |   √   |      |       trpo        |
|               PPO               |    √     |     √      |   √   |      |        ppo        |
|               DQN               |    √     |            |   √   |  √   |        dqn        |
|           Double DQN            |    √     |            |   √   |  √   |       ddqn        |
|       Dueling Double DQN        |    √     |            |   √   |  √   |       dddqn       |
|          Averaged DQN           |    √     |            |   √   |  √   |    averaged_dqn   |
|        Bootstrapped DQN         |    √     |            |   √   |  √   |  bootstrappeddqn  |
|         Soft Q-Learning         |    √     |            |   √   |  √   |        sql        |
|               C51               |    √     |            |   √   |  √   |        c51        |
|             QR-DQN              |    √     |            |   √   |  √   |       qrdqn       |
|               IQN               |    √     |            |   √   |  √   |        iqn        |
|             Rainbow             |    √     |            |   √   |  √   |      rainbow      |
|               DPG               |    √     |     √      |   √   |  √   |        dpg        |
|              DDPG               |    √     |     √      |   √   |  √   |       ddpg        |
|               TD3               |    √     |     √      |   √   |  √   |        td3        |
|       SAC(has V network)        |    √     |     √      |   √   |  √   |       sac_v       |
|               SAC               |    √     |     √      |   √   |  √   |        sac        |
|               TAC               |   sac    |     √      |   √   |  √   |        tac        |
|             MaxSQN              |    √     |            |   √   |  √   |      maxsqn       |
|               OC                |    √     |     √      |   √   |  √   |        oc         |
|               AOC               |    √     |     √      |   √   |  √   |        aoc        |
|              PPOC               |    √     |     √      |   √   |  √   |       ppoc        |
|               IOC               |    √     |     √      |   √   |  √   |        ioc        |
|              HIRO               |    √     |     √      |       |      |       hiro        |
|              CURL               |    √     |     √      |   √   |      |       curl        |
|               VDN               |    √     |            |   √   |      |        vdn        |
|             MADDPG              |    √     |     √      |   √   |      |      maddpg       |

## Getting started

```python
"""
usage: run.py [-h] [-c COPYS] [--seed SEED] [-r] [-p {gym,unity}]
              [-a {pg,trpo,ppo,a2c,cem,aoc,ppoc,qs,ac,dpg,ddpg,td3,sac_v,sac,tac,dqn,ddqn,dddqn,averaged_dqn,c51,qrdqn,rainbow,iqn,maxsqn,sql,bootstrappeddqn,curl,oc,ioc,hiro,maddpg,vdn}]
              [-i] [-l LOAD_PATH] [-m MODELS] [-n NAME] [-s SAVE_FREQUENCY]
              [--apex {learner,worker,buffer,evaluator}] [--config-file CONFIG_FILE] [--store-dir STORE_DIR]
              [--episode-length EPISODE_LENGTH] [--prefill-steps PREFILL_STEPS] [--prefill-choose] [--hostname]
              [--info INFO] [-e ENV_NAME] [-f FILE_NAME] [--no-save] [-d DEVICE]

optional arguments:
  -h, --help            show this help message and exit
  -c COPYS, --copys COPYS
                        nums of environment copys that collect data in parallel
  --seed SEED           specify the random seed of module random, numpy and pytorch
  -r, --render          whether render game interface
  -p {gym,unity}, --platform {gym,unity}
                        specify the platform of training environment
  -a {pg,trpo,ppo,a2c,cem,aoc,ppoc,qs,ac,dpg,ddpg,td3,sac_v,sac,tac,dqn,ddqn,dddqn,averaged_dqn,c51,qrdqn,rainbow,iqn,maxsqn,sql,bootstrappeddqn,curl,oc,ioc,hiro,maddpg,vdn}, --algorithm {pg,trpo,ppo,a2c,cem,aoc,ppoc,qs,ac,dpg,ddpg,td3,sac_v,sac,tac,dqn,ddqn,dddqn,averaged_dqn,c51,qrdqn,rainbow,iqn,maxsqn,sql,bootstrappeddqn,curl,oc,ioc,hiro,maddpg,vdn}
                        specify the training algorithm
  -i, --inference       inference the trained model, not train policies
  -l LOAD_PATH, --load-path LOAD_PATH
                        specify the name of pre-trained model that need to load
  -m MODELS, --models MODELS
                        specify the number of trails that using different random seeds
  -n NAME, --name NAME  specify the name of this training task
  -s SAVE_FREQUENCY, --save-frequency SAVE_FREQUENCY
                        specify the interval that saving model checkpoint
  --apex {learner,worker,buffer,evaluator}
  --config-file CONFIG_FILE
                        specify the path of training configuration file
  --store-dir STORE_DIR
                        specify the directory that store model, log and others
  --episode-length EPISODE_LENGTH
                        specify the maximum step per episode
  --prefill-steps PREFILL_STEPS
                        specify the number of experiences that should be collected before start training, use for
                        off-policy algorithms
  --prefill-choose      whether choose action using model or choose randomly
  --hostname            whether concatenate hostname with the training name
  --info INFO           write another information that describe this training task
  -e ENV_NAME, --env-name ENV_NAME
                        specify the environment name
  -f FILE_NAME, --file-name FILE_NAME
                        specify the path of builded training environment of UNITY3D
  --no-save             specify whether save models/logs/summaries while training or not
  -d DEVICE, --device DEVICE
                        specify the device that operate Torch.Tensor
```

```python
"""
Example:
    python run.py
    python run.py --config-file rls/configs/examples/gym_config.yaml
    python run.py -p gym -a dqn -e CartPole-v0 -c 12 -n dqn_cartpole --no-save
    python run.py -p unity -a ppo -n run_with_unity
    python run.py -p unity --file-name /root/env/3dball.app -a sac -n run_with_execution_file
"""
```

## Notes

1. record directory format is `Platform/Environment/Algorithm/Training name/log&model`
2. multi-agents algorithms doesn't support PER for now
3. **need 3 steps to implement a new algorithm**
    1. **policy** write `.py` in `rls/algos/{single/multi/hierarchical}` directory and make the policy inherit from class `Policy`, `On_Policy`, `Off_Policy` or other super-class defined in `rls/algos/base`
    2. **config** write default configuration in `rls/configs/algorithms.yaml`
    3. **register** register new algorithm at dictionary *algos* in `rls/algos/__init__.py`, make sure the class name matches the name of the algorithm class
4. set algorithms' hyper-parameters in [rls/configs/algorithms.yaml](https://github.com/StepNeverStop/RLs/blob/master/rls/configs/algorithms.yaml)
5. set training default configuration in [rls/configs/train.yaml](https://github.com/StepNeverStop/RLs/blob/master/rls/configs/train.yaml)
6. change neural network structure in [rls/nn/models.py](https://github.com/StepNeverStop/RLs/blob/master/rls/nn/models.py)

## Ongoing things

- DARQN
- ACER
- Ape-X
- R2D2
- ~~ACKTR~~

## Giving credit

If using this repository for your research, please cite:
```
@misc{RLs,
  author = {Keavnn},
  title = {RLs: Reinforcement Learning research framework for Unity3D and Gym},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/StepNeverStop/RLs}},
}
```

## Issues

Any questions/errors about this project, please let me know in [here](https://github.com/StepNeverStop/RLs/issues/new).
