#!/usr/bin/env python3
# encoding: utf-8

import os
import re
import gym

from typing import Dict
from collections import defaultdict

from rls.common.yaml_ops import load_config
from rls.envs.gym.wrappers.wrappers import *


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


def make_atari(env,
               max_episode_steps=None,
               noop_max=30,
               skip=4,
               deepmind_config=dict(episode_life=True,
                                    resize=True,
                                    grayscale=True,
                                    width=84,
                                    height=84,
                                    scale=False,
                                    clip_rewards=True,
                                    frame_stack=True,
                                    stack=4)):
    env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=skip)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=int(max_episode_steps))

    env = wrap_deepmind(env, **deepmind_config)
    return env


def wrap_deepmind(env,
                  episode_life=True,
                  resize=True,
                  grayscale=True,
                  width=84,
                  height=84,
                  scale=False,
                  clip_rewards=True,
                  frame_stack=True,
                  stack=4):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = GrayResizeEnv(env, resize=resize, grayscale=grayscale,
                        width=width, height=height)
    if scale:
        env = ScaleEnv(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = StackEnv(env, stack=stack)
    return env


def make_env(index: int = 0,

             env_name: str = 'CartPole-v0',
             action_skip=False,
             skip=4,
             obs_stack=False,
             stack=4,
             noop=False,
             noop_max=30,
             obs_grayscale=False,
             obs_resize=False,
             resize=[84, 84],
             obs_scale=False,
             max_episode_steps=None,
             atari_config=dict(max_episode_steps=None,
                               noop_max=30,
                               skip=4,
                               deepmind_config=dict(episode_life=True,
                                                    resize=True,
                                                    grayscale=True,
                                                    width=84,
                                                    height=84,
                                                    scale=False,
                                                    clip_rewards=True,
                                                    frame_stack=True,
                                                    stack=4)),
             donkey_config=dict(port=9091,
                                car_name='Agent',
                                exe_path='...\donkey_sim.exe',
                                host='127.0.0.1',
                                body_style='donkey',
                                body_rgb=[128, 128, 128],
                                font_size=100,
                                racer_name='test',
                                country='USA',
                                bio='I am test client',
                                max_cte=20),
             **kwargs):
    env_type = get_env_type(env_name)

    env_params = {}
    if env_type == 'pybullet_envs.bullet':
        env_params.update({'renders': kwargs.get('inference', False)})

    elif env_type == 'gym_donkeycar.envs.donkey_env':
        import uuid
        # [120, 160, 3]
        donkey_config['port'] += index
        donkey_config['car_name'] += str(index)
        donkey_config['guid'] = str(uuid.uuid4())
        env_params['conf'] = donkey_config

    env = gym.make(env_name, **env_params)
    env = BaseEnv(env)

    if env_type == 'gym.envs.atari':
        assert 'NoFrameskip' in env.spec.id, 'env id should contain NoFrameskip.'
        env = make_atari(env, **atari_config)
    else:
        if env_name.split('-')[0] == 'MiniGrid':
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
