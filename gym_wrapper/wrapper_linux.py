import gym
import ray
import numpy as np
from typing import Dict
from gym.spaces import Box, Discrete, Tuple
from .wrappers import SkipEnv, StackEnv, GrayResizeEnv, ScaleEnv, OneHotObsEnv, BoxActEnv, BaseEnv


@ray.remote
class RayEnv:
    def __init__(self, env_func, kwargs):
        self.env = env_func(**kwargs)

    def seed(self, s):
        self.env.seed(s)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def sample(self):
        return self.env.action_sample()


class gym_envs(object):

    def __init__(self, kwargs: Dict):
        '''
        Input:
            gym_env_name: gym training environment id, i.e. CartPole-v0
            n: environment number
            render_mode: mode of rendering, optional: first, last, all, random_[num] -> i.e. random_2, [list] -> i.e. [0, 2, 4]
        '''
        ray.init()

        self.n = kwargs['env_num']  # environments number
        seed = kwargs['env_seed']
        render_mode = kwargs.get('render_mode', 'first')

        def get_env(kwargs):
            gym_env_name = kwargs['env_name']
            action_skip = bool(kwargs.get('action_skip', False))
            skip = int(kwargs.get('skip', 4))
            obs_stack = bool(kwargs.get('obs_stack', False))
            stack = int(kwargs.get('stack', 4))

            noop = bool(kwargs.get('noop', False))
            noop_max = int(kwargs.get('noop_max', 30))
            obs_grayscale = bool(kwargs.get('obs_grayscale', False))
            obs_resize = bool(kwargs.get('obs_resize', False))
            resize = kwargs.get('resize', [84, 84])
            obs_scale = bool(kwargs.get('obs_scale', False))

            env = gym.make(gym_env_name)
            env = BaseEnv(env)
            if noop and isinstance(env.observation_space, Box) and len(env.observation_space.shape) == 3:
                env = NoopResetEnv(env, noop_max=noop_max)
            if action_skip:
                env = SkipEnv(env, skip=skip)
            if isinstance(env.observation_space, Box):
                if len(env.observation_space.shape) == 3:
                    if obs_grayscale or obs_resize:
                        env = GrayResizeEnv(env, resize=obs_resize, grayscale=obs_grayscale, width=resize[0], height=resize[-1])
                    if obs_scale:
                        env = ScaleEnv(env)
                if obs_stack:
                    env = StackEnv(env, stack=stack)
            else:
                env = OneHotObsEnv(env)

            if isinstance(env.action_space, Box) and len(env.action_space.shape) == 1:
                env = BoxActEnv(env)
            return env

        self._initialize(
            env=get_env(kwargs)
        )
        self.envs = [RayEnv.remote(get_env, kwargs) for i in range(self.n)]
        self.seeds = [seed + i for i in range(self.n)]
        [env.seed.remote(s) for env, s in zip(self.envs, self.seeds)]
        self._get_render_index(render_mode)

    def _initialize(self, env):
        assert isinstance(env.observation_space, (Box, Discrete)) and isinstance(env.action_space, (Box, Discrete)), 'action_space and observation_space must be one of available_type'
        # process observation
        ObsSpace = env.observation_space
        if isinstance(ObsSpace, Box):
            self.s_dim = ObsSpace.shape[0] if len(ObsSpace.shape) == 1 else 0
            self.obs_high = ObsSpace.high
            self.obs_low = ObsSpace.low
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
            self.a_dim_or_list = ActSpace.shape
        elif isinstance(ActSpace, Tuple):
            assert all([isinstance(i, Discrete) for i in ActSpace]) == True, 'if action space is Tuple, each item in it must have type Discrete'
            self.action_type = 'Tuple(Discrete)'
            self._is_continuous = False
            self.a_dim_or_list = [i.n for i in ActSpace]
        else:
            self.action_type = 'discrete'
            self._is_continuous = False
            self.a_dim_or_list = [env.action_space.n]

        self.reward_threshold = env.env.spec.reward_threshold  # reward threshold refer to solved
        env.close()

    @property
    def is_continuous(self):
        return self._is_continuous

    def _get_render_index(self, render_mode):
        '''
        get render windows list, i.e. [0, 1] when there are 4 training enviornment.
        render_mode: mode of rendering, 
        optional: 
            - first, 
            - last, 
            - all, 
            - random_[num] -> i.e. random_2, 
            - [list] -> i.e. [0, 2, 4]
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

    def reset(self):
        self.dones_index = []
        obs = np.asarray(ray.get([env.reset.remote() for env in self.envs]))
        return obs

    def partial_reset(self):
        obs = np.asarray(ray.get([self.envs[i].reset.remote() for i in self.dones_index]))
        return obs

    def render(self):
        '''
        render game windows.
        '''
        [self.envs[i].render.remote() for i in self.render_index]

    def sample_actions(self):
        '''
        generate random actions for all training environment.
        '''
        return np.asarray(ray.get([env.sample.remote() for env in self.envs]))

    def step(self, actions):
        if self.action_type == 'discrete':
            actions = actions.reshape(-1,)
        elif self.action_type == 'Tuple(Discrete)':
            actions = actions.reshape(self.n, -1).tolist()
        obs, reward, done, info = list(zip(*ray.get([env.step.remote(action) for env, action in zip(self.envs, actions)])))
        self.dones_index = np.where(done)[0]
        return (np.asarray(obs),
                np.asarray(reward),
                np.asarray(done),
                info)

    def close(self):
        '''
        close all environments.
        '''
        ray.get([env.close.remote() for env in self.envs])
        ray.shutdown()
