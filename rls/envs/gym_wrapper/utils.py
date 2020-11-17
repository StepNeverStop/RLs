#!/usr/bin/env python3
# encoding: utf-8

import os
import re
import gym

from typing import Dict
from collections import defaultdict

from rls.common.yaml_ops import load_yaml
from rls.envs.gym_wrapper.wrappers import *


def get_env_type(env_id):

    _game_envs = defaultdict(set)
    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    env_type = None
    for g, e in _game_envs.items():
        if env_id in e:
            env_type = g
            break
    if ':' in env_id:
        env_type = re.sub(r':.*', '', env_id)
    assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type


def make_atari(env, config: Dict):
    max_episode_steps = config.get('max_episode_steps', None)
    env = NoopResetEnv(env, noop_max=int(config.get('noop_max', 30)))
    env = MaxAndSkipEnv(env, skip=int(config.get('skip', 4)))
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=int(max_episode_steps))

    env = wrap_deepmind(env, config)
    return env


def wrap_deepmind(env, config: Dict):
    """Configure environment for DeepMind-style Atari.
    """
    if bool(config.get('episode_life', True)):
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = GrayResizeEnv(env, resize=bool(config.get('resize', True)), grayscale=bool(config.get('grayscale', True)),
                        width=int(config.get('width', 84)), height=int(config.get('height', 84)))
    if bool(config.get('scale', False)):
        env = ScaleEnv(env)
    if bool(config.get('clip_rewards', True)):
        env = ClipRewardEnv(env)
    if bool(config.get('frame_stack', True)):
        env = StackEnv(env, stack=int(config.get('stack', 4)))
    return env


def build_env(config: Dict, index: int = 0):
    gym_env_name = config['env_name']

    env_type = get_env_type(gym_env_name)

    env_params = {}
    if env_type == 'pybullet_envs.bullet':
        env_params.update({'renders': bool(config.get('inference', False))})

    elif env_type == 'gym_donkeycar.envs.donkey_env':
        _donkey_conf = load_yaml(f'{os.path.dirname(__file__)}/config.yaml')['donkey']
        import uuid
        # [120, 160, 3]
        _donkey_conf['port'] += index
        _donkey_conf['car_name'] += str(index)
        _donkey_conf['guid'] = str(uuid.uuid4())
        env_params['conf'] = _donkey_conf

    env = gym.make(gym_env_name, **env_params)
    env = BaseEnv(env)

    if env_type == 'gym.envs.atari':
        assert 'NoFrameskip' in env.spec.id, 'env id should contain NoFrameskip.'

        default_config = load_yaml(f'{os.path.dirname(__file__)}/config.yaml')['atari']
        env = make_atari(env, default_config)
    else:
        action_skip = bool(config.get('action_skip', False))
        skip = int(config.get('skip', 4))
        obs_stack = bool(config.get('obs_stack', False))
        stack = int(config.get('stack', 4))
        noop = bool(config.get('noop', False))
        noop_max = int(config.get('noop_max', 30))
        obs_grayscale = bool(config.get('obs_grayscale', False))
        obs_resize = bool(config.get('obs_resize', False))
        resize = config.get('resize', [84, 84])
        obs_scale = bool(config.get('obs_scale', False))
        max_episode_steps = config.get('max_episode_steps', None)

        if gym_env_name.split('-')[0] == 'MiniGrid':
            env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)  # Get pixel observations, or RGBImgObsWrapper
            env = gym_minigrid.wrappers.ImgObsWrapper(env)  # Get rid of the 'mission' field
        if noop and isinstance(env.observation_space, Box) and len(env.observation_space.shape) == 3:
            env = NoopResetEnv(env, noop_max=noop_max)
        if action_skip:
            env = MaxAndSkipEnv(env, skip=4)
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
        env = TimeLimit(env, max_episode_steps)

    if isinstance(env.action_space, Box) and len(env.action_space.shape) == 1:
        env = BoxActEnv(env)

    if not (isinstance(env.observation_space, Box) and len(env.observation_space.shape) == 3):
        env = DtypeEnv(env)
    return env
