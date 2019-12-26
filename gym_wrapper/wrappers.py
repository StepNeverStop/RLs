
import gym
import cv2
cv2.ocl.setUseOpenCL(False)
import numpy as np
from collections import deque
from gym.spaces import Box, Discrete, Tuple


class BaseEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def action_sample(self):
        return self.env.action_space.sample()


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """ From baselines
        Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


class SkipEnv(gym.Wrapper):
    '''
    底层，因为这样可以减少对状态的数据处理
    '''

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        reward = 0.
        for _ in range(self._skip):
            obs, r, done, info = self.env.step(action)
            reward += r
            if done:
                break
        return obs, reward, done, info


class StackEnv(gym.Wrapper):
    '''
    上层，因为这样不必在得到状态后还需要对其进行处理
    '''

    def __init__(self, env, stack=4):
        super().__init__(env)
        self._stack = stack
        self.obss = deque([], maxlen=self._stack)
        assert isinstance(env.observation_space, Box)
        if len(env.observation_space.shape) == 1 or len(env.observation_space.shape) == 3:
            low = np.tile(env.observation_space.low, stack)
            high = np.tile(env.observation_space.high, stack)
        self.observation_space = Box(low=low,
                                     high=high,
                                     dtype=env.observation_space.dtype)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        for _ in range(self._stack):
            self.obss.append(obs)
        return LazyFrames(list(self.obss))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.obss.append(obs)
        return LazyFrames(list(self.obss)), reward, done, info


class GrayResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, resize=True, grayscale=True, *, width=84, height=84):
        super().__init__(env)
        self._resize = resize
        if resize:
            self._width = width
            self._height = height
        else:
            shp = env.observation_space.shape
            assert len(shp) == 3
            self._width = shp[0]
            self._height = shp[1]
        self._grayscale = grayscale
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )

    def observation(self, obs):
        if self._grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        if self._resize:
            obs = cv2.resize(
                obs, (self._width, self._height), interpolation=cv2.INTER_AREA
            )
        if self._grayscale:
            obs = np.expand_dims(obs, -1)
        return obs


class ScaleEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class OneHotObsEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        sh = env.observation_space
        if isinstance(sh.n, (int, np.int32)):
            obs_dim = [int(sh.n)]
            self._obs_dim = (1,)
        else:
            obs_dim = list(sh.n)    # 在CliffWalking-v0环境其类型为numpy.int32
            self._obs_dim = obs_dim
        self.one_hot_len = np.array(obs_dim).prod()
        self.multiplication_factor = np.asarray(obs_dim[1:] + [1])

    def observation(self, obs):
        return self._one_hot(obs)

    def _one_hot(self, obs):
        """
        Change discrete observation from list(int) to list(one_hot) format.
        for example:
            action: [1, 0]
            observation space: [3, 4]
            then, output: [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
        """
        obs = np.reshape(obs, self._obs_dim)
        ints = np.dot(obs, self.multiplication_factor)
        obs = np.zeros([self.one_hot_len, ])
        obs[ints] = 1.
        return obs


class BoxActEnv(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        asp = env.action_space
        assert isinstance(asp, Box) and len(asp.shape) == 1
        self._mu = (asp.high + asp.low) / 2
        self._sigma = (asp.high - asp.low) / 2

    def action_sample(self):
        a = self.env.action_space.sample()
        return (self.env.action_space.sample() - self._mu) / self._sigma

    def step(self, action):
        return self.env.step(self.action(action))

    def action(self, action):
        return self._sigma * action + self._mu


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

    def __repr__(self):
        return self._force()[:]
