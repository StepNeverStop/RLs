import gym
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('envs.wrappers.gym_wrapper.gym_env')

try:
    import gym_minigrid
except:
    logger.warning("import gym_minigrid failed, using 'pip3 install gym-minigrid' install it.")
    pass

try:
    # if wanna render, added 'renders=True' or(depends on env) 'render=True' in gym.make() function manually.
    import pybullet_envs
except:
    logger.warning("import pybullet_envs failed, using 'pip3 install PyBullet' install it.")
    pass

import numpy as np
from typing import Dict
from copy import deepcopy
from utils.sth import sth
from gym.spaces import Box, Discrete, Tuple
from envs.wrappers.gym_wrapper.utils import build_env

import platform
use_ray = False
if platform.system() != "Windows" and use_ray:
    from . import ray_wrapper as Asyn
else:
    from . import threading_wrapper as Asyn


class gym_envs(object):

    def __init__(self, config: Dict):
        '''
        Input:
            gym_env_name: gym training environment id, i.e. CartPole-v0
            n: environment number
            render_mode: mode of rendering, optional: first, last, all, random_[num] -> i.e. random_2, [list] -> i.e. [0, 2, 4]
        '''
        self.n = config['env_num']  # environments number
        render_mode = config.get('render_mode', 'first')

        self.info_env = build_env(config)
        self._initialize(env=self.info_env)
        self.info_env.close()
        del self.info_env

        self.envs = Asyn.init_envs(build_env, config, self.n, config['env_seed'])
        self._get_render_index(render_mode)

    def _initialize(self, env):
        assert isinstance(env.observation_space, (Box, Discrete)) and isinstance(env.action_space, (Box, Discrete)), 'action_space and observation_space must be one of available_type'
        # process observation
        ObsSpace = env.observation_space
        if isinstance(ObsSpace, Box):
            self.s_dim = ObsSpace.shape[0] if len(ObsSpace.shape) == 1 else 0
            # self.obs_high = ObsSpace.high
            # self.obs_low = ObsSpace.low
        else:
            self.s_dim = int(ObsSpace.n)
        if len(ObsSpace.shape) == 3:
            self.obs_type = 'visual'
            self.visual_sources = 1
            self.visual_resolution = list(ObsSpace.shape)
        else:
            self.obs_type = 'vector'
            self.visual_sources = 0
            self.visual_resolution = []

        # process action
        ActSpace = env.action_space
        if isinstance(ActSpace, Box):
            assert len(ActSpace.shape) == 1, 'if action space is continuous, the shape length of action must equal to 1'
            self.action_type = 'continuous'
            self._is_continuous = True
            self.a_dim = ActSpace.shape[0]
        elif isinstance(ActSpace, Tuple):
            assert all([isinstance(i, Discrete) for i in ActSpace]) == True, 'if action space is Tuple, each item in it must have type Discrete'
            self.action_type = 'Tuple(Discrete)'
            self._is_continuous = False
            self.a_dim = int(np.asarray([i.n for i in ActSpace]).prod())
            self.discrete_action_dim_list = [i.n for i in ActSpace]
        else:
            self.action_type = 'discrete'
            self._is_continuous = False
            self.a_dim = env.action_space.n
            self.discrete_action_dim_list = [env.action_space.n]

        self.reward_threshold = env.env.spec.reward_threshold  # reward threshold refer to solved

    @property
    def is_continuous(self):
        return self._is_continuous

    def _get_render_index(self, render_mode):
        '''
        get render windows list, i.e. [0, 1] when there are 4 training enviornment.
        '''
        assert isinstance(render_mode, (list, str)), 'render_mode must have type of str or list.'
        if isinstance(render_mode, list):
            assert all([isinstance(i, int) for i in render_mode]), 'items in render list must have type of int'
            assert min(index) >= 0, 'index must larger than zero'
            assert max(index) <= self.n, 'render index cannot larger than environment number.'
            self.render_index = render_mode
        elif isinstance(render_mode, str):
            if render_mode == 'first':
                self.render_index = [0]
            elif render_mode == 'last':
                self.render_index = [-1]
            elif render_mode == 'all':
                self.render_index = [i for i in range(self.n)]
            else:
                a, b = render_mode.split('_')
                if a == 'random' and 0 < int(b) <= self.n:
                    import random
                    self.render_index = random.sample([i for i in range(self.n)], int(b))
        else:
            raise Exception('render_mode must be first, last, all, [list] or random_[num]')

    def render(self, record=False):
        '''
        render game windows.
        '''
        Asyn.op_func([self.envs[i] for i in self.render_index], Asyn.OP.RENDER)

    def close(self):
        '''
        close all environments.
        '''
        Asyn.op_func(self.envs, Asyn.OP.CLOSE)

    def sample_actions(self):
        '''
        generate random actions for all training environment.
        '''
        return np.asarray(Asyn.op_func(self.envs, Asyn.OP.SAMPLE))

    def reset(self):
        obs = np.asarray(Asyn.op_func(self.envs, Asyn.OP.RESET))
        if self.obs_type == 'visual':
            obs = obs[:, np.newaxis, ...]
        return obs

    def step(self, actions):
        actions = np.array(actions)
        if not self.is_continuous:
            actions = sth.int2action_index(actions, self.discrete_action_dim_list)
            if self.action_type == 'discrete':
                actions = actions.reshape(-1,)
            elif self.action_type == 'Tuple(Discrete)':
                actions = actions.reshape(self.n, -1).tolist()
        results = Asyn.op_func(self.envs, Asyn.OP.STEP, actions)
        obs, reward, done, info = [np.asarray(e) for e in zip(*results)]
        reward = reward.astype('float32')
        dones_index = np.where(done)[0]
        if dones_index.shape[0] > 0:
            correct_new_obs = self.partial_reset(obs, dones_index)
        else:
            correct_new_obs = obs
        if self.obs_type == 'visual':
            obs = obs[:, np.newaxis, ...]
            correct_new_obs = correct_new_obs[:, np.newaxis, ...]
        return obs, reward, done, info, correct_new_obs

    def partial_reset(self, obs, dones_index):
        correct_new_obs = deepcopy(obs)
        partial_obs = np.asarray(Asyn.op_func([self.envs[i] for i in dones_index], Asyn.OP.RESET))
        correct_new_obs[dones_index] = partial_obs
        return correct_new_obs
