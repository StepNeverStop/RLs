#!/usr/bin/env python3
# encoding: utf-8

from typing import Dict
from copy import deepcopy

from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


def make_env(env_kargs: Dict):
    logger.info('Initialize environment begin...')
    if env_kargs['type'] == 'gym':
        env = make_gym_env(env_kargs)
    elif env_kargs['type'] == 'unity':
        env = make_unity_env(env_kargs)
    else:
        raise Exception('Unknown environment type.')
    logger.info('Initialize environment successful.')
    return env


def make_gym_env(env_kargs: Dict):
    from rls.envs.gym_env import gym_envs

    copied_env_kargs = deepcopy(env_kargs)
    env = gym_envs(copied_env_kargs)
    return env


def make_unity_env(env_kargs: Dict):
    from rls.envs.unity_wrapper import (BasicUnityEnvironment,
                                        GrayVisualWrapper,
                                        ResizeVisualWrapper,
                                        ScaleVisualWrapper,
                                        BasicActionWrapper,
                                        StackVisualWrapper)

    copied_env_kargs = deepcopy(env_kargs)
    env = BasicUnityEnvironment(copied_env_kargs)
    logger.debug('Unity BasicUnityEnvironment success.')

    if copied_env_kargs['obs_grayscale']:
        env = GrayVisualWrapper(env)

    if copied_env_kargs['obs_resize']:
        env = ResizeVisualWrapper(env, resize=copied_env_kargs['resize'])

    if copied_env_kargs['obs_scale']:
        env = ScaleVisualWrapper(env)

    if copied_env_kargs['obs_stack']:
        env = StackVisualWrapper(env, stack_nums=env_kargs['stack_visual_nums'])

    env = BasicActionWrapper(env)
    logger.debug('Unity BasicActionWrapper success.')

    return env
