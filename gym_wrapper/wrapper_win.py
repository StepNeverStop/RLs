import gym
import numpy as np
import threading
from gym.spaces import Box, Discrete, Tuple
import cv2
cv2.ocl.setUseOpenCL(False)
from collections import deque


class FakeMultiThread(threading.Thread):

    def __init__(self, func, args=()):
        super().__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

class StackAndGrey(gym.Wrapper):
    # TODO: skip, stack, grey, resize for atari
    def __init__(self, env):
        super().__init__(env)
        self.k = 4
        self.obss = deque([], maxlen=self.k)
    
    def grey_resize(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(
            obs, (84, 84), interpolation=cv2.INTER_AREA
        )
        obs = np.expand_dims(obs, -1)
        return obs
            
    def reset(self):
        obs = self.env.reset()
        obs = self.grey_resize(obs)
        for _ in range(self.k):
            self.obss.append(obs)
        return LazyFrames(list(self.obss))

    def step(self, action):
        r = 0.
        for _ in range(4):
            obs, reward, done, info = self.env.step(action)
            r += reward
            if done:
                break
        obs = self.grey_resize(obs)
        self.obss.append(obs)
        return LazyFrames(list(self.obss)), r, done, info

class LazyFrames(object):
    '''
    stole this from baselines.
    '''
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]

class gym_envs(object):

    def __init__(self, gym_env_name, n, seed=0, render_mode='first'):
        '''
        Input:
            gym_env_name: gym training environment id, i.e. CartPole-v0
            n: environment number
            render_mode: mode of rendering, optional: first, last, all, random_[num] -> i.e. random_2, [list] -> i.e. [0, 2, 4]
        '''
        self.n = n  # environments number
        self._initialize(gym_env_name)
        self.envs = [gym.make(gym_env_name) for _ in range(self.n)]
        # self.envs = [StackAndGrey(env) for env in self.envs]
        self.seeds = [seed + i for i in range(n)]
        [env.seed(s) for env, s in zip(self.envs, self.seeds)]
        self._get_render_index(render_mode)

    def _initialize(self, gym_env_name):
        env = gym.make(gym_env_name)
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

        if hasattr(ObsSpace, 'n'):
            self.toOneHot = True
            if isinstance(ObsSpace.n, (int, np.int32)):
                obs_dim = [int(ObsSpace.n)]
            else:
                obs_dim = list(ObsSpace.n)    # 在CliffWalking-v0环境其类型为numpy.int32
            self.multiplication_factor = np.asarray(obs_dim[1:] + [1])
            self.one_hot_len = self.multiplication_factor.prod()
        else:
            self.toOneHot = False

        # process action
        ActSpace = env.action_space
        if isinstance(ActSpace, Box):
            assert len(ActSpace.shape) == 1, 'if action space is continuous, the shape length of action must equal to 1'
            self.action_type = 'continuous'
            self.action_high = ActSpace.high
            self.action_low = ActSpace.low
            self.a_dim_or_list = ActSpace.shape
        elif isinstance(ActSpace, Tuple):
            assert all([isinstance(i, Discrete) for i in ActSpace]) == True, 'if action space is Tuple, each item in it must have type Discrete'
            self.action_type = 'Tuple(Discrete)'
            self.a_dim_or_list = [i.n for i in ActSpace]
        else:
            self.action_type = 'discrete'
            self.a_dim_or_list = [env.action_space.n]
        self.action_mu, self.action_sigma = self._get_action_normalize_factor()

        self.reward_threshold = env.env.spec.reward_threshold  # reward threshold refer to solved
        env.close()

    @property
    def is_continuous(self):
        return True if self.action_type == 'continuous' else False

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

    def render(self):
        '''
        render game windows.
        '''
        [self.envs[i].render() for i in self.render_index]

    def close(self):
        '''
        close all environments.
        '''
        [env.close() for env in self.envs]

    def sample_action(self):
        '''
        generate ramdom actions for all training environment.
        '''
        return np.asarray([env.action_space.sample() for env in self.envs])

    def reset(self):
        self.dones_index = []
        threadpool = []
        for i in range(self.n):
            th = FakeMultiThread(self.envs[i].reset, args=())
            threadpool.append(th)
        for th in threadpool:
            th.start()
        for th in threadpool:
            threading.Thread.join(th)
        obs = np.asarray([threadpool[i].get_result() for i in range(self.n)])
        obs = self._maybe_one_hot(obs)
        return obs

    def step(self, actions, scale=True):
        if scale == True:
            actions = self.action_sigma * actions + self.action_mu
        if self.action_type == 'discrete':
            actions = actions.reshape(-1,)
        elif self.action_type == 'Tuple(Discrete)':
            actions = actions.reshape(self.n, -1).tolist()
        threadpool = []
        for i in range(self.n):
            th = FakeMultiThread(self.envs[i].step, args=(actions[i], ))
            threadpool.append(th)
        for th in threadpool:
            th.start()
        for th in threadpool:
            threading.Thread.join(th)
        results = [threadpool[i].get_result() for i in range(self.n)]

        obs, reward, done, info = [np.asarray(e) for e in zip(*results)]
        obs = self._maybe_one_hot(obs)
        self.dones_index = np.where(done)[0]
        return obs, reward, done, info

    def partial_reset(self):
        threadpool = []
        for i in self.dones_index:
            th = FakeMultiThread(self.envs[i].reset, args=())
            threadpool.append(th)
        for th in threadpool:
            th.start()
        for th in threadpool:
            threading.Thread.join(th)
        obs = np.asarray([threadpool[i].get_result() for i in range(self.dones_index.shape[0])])
        obs = self._maybe_one_hot(obs, is_partial=True)
        return obs

    def _get_action_normalize_factor(self):
        '''
        get action mu and sigma. mu: action bias. sigma: action scale
        input: 
            self.action_low: [-2, -3],
            self.action_high: [2, 6]
        return: 
            mu: [0, 1.5], 
            sigma: [2, 4.5]
        '''
        if self.action_type == 'continuous':
            return (self.action_high + self.action_low) / 2, (self.action_high - self.action_low) / 2
        else:
            return 0, 1

    def _maybe_one_hot(self, obs, is_partial=False):
        """
        Change discrete observation from list(int) to list(one_hot) format.
        for example:
            action: [[1, 0], [2, 1]]
            observation space: [3, 4]
            environment number: 2
            then, output: [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
                           [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
        """
        if self.toOneHot:
            obs_number = len(self.dones_index) if is_partial else self.n
            obs = obs.reshape(obs_number, -1)
            ints = obs.dot(self.multiplication_factor)
            x = np.zeros([obs.shape[0], self.one_hot_len])
            for i, j in enumerate(ints):
                x[i, j] = 1
            return x
        else:
            return obs
