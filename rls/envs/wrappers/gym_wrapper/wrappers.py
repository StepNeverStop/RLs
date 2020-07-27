#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('envs.wrappers.gym_wrapper.wrappers')
try:
    import cv2
    cv2.ocl.setUseOpenCL(False)
except:
    logger.warning('opencv-python is needed to train visual-based model.')
    pass

try:
    import imageio
except:
    logger.warning('imageio should be installed to record vedio if needed.')
    pass

import numpy as np

from collections import deque
from gym.spaces import \
    Box, \
    Discrete, \
    Tuple

from rls.envs.wrappers.LazyFrames import LazyFrames


class BaseEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def action_sample(self):
        return self.env.action_space.sample()

    def render(self, mode, **kwargs):
        filename = kwargs.get('filename', None)
        fps = kwargs.get('fps', 30)
        if filename is not None:
            if not hasattr(self, 'video_writer'):
                self.video_writer = imageio.get_writer(filename, fps=fps)
            self.video_writer.append_data(self.env.render(mode='rgb_array'))
        else:
            self.env.render(mode='human')

    def close(self):
        if hasattr(self, 'video_writer'):
            self.video_writer.close()
        self.env.close()


class NoopResetEnv(gym.Wrapper):
    '''Execute no-op actions before take real actions.'''

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


class OneHotObsEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        sh = env.observation_space
        if hasattr(sh, '__len__'):
            obs_dim = list(sh.n)    # 在CliffWalking-v0环境其类型为numpy.int32
            self._obs_dim = obs_dim
        else:
            obs_dim = [int(sh.n)]
            self._obs_dim = (1,)
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


class TimeLimit(gym.Wrapper):
    '''Set max episode steps for tasks.'''

    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = self.env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps or 10000
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class DtypeEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return np.array(obs, dtype=np.float32)


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        """
        clips the reward to {+1, 0, -1} by its sign.
        :param env: (Gym Environment) the environment
        """
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """
        Bin reward to {+1, 0, -1} by its sign.
        :param reward: (float)
        """
        return np.sign(reward)


class MaxAndSkipEnv(gym.Wrapper):
    '''
    Execute same action several times.
    底层，因为这样可以减少对状态的数据处理
    '''

    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)
