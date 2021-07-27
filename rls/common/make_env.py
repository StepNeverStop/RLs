#!/usr/bin/env python3
# encoding: utf-8

from typing import Dict
from copy import deepcopy

from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


def make_env(env_kargs: Dict):
    logger.info('Initialize environment begin...')
    if env_kargs['platform'] == 'gym':
        env = make_gym_env(env_kargs)
    elif env_kargs['platform'] == 'unity':
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
                                        ScaleVisualWrapper)

    copied_env_kargs = deepcopy(env_kargs)
    env = BasicUnityEnvironment(copied_env_kargs)
    logger.debug('Unity BasicUnityEnvironment success.')

    if copied_env_kargs['obs_scale']:
        env = ScaleVisualWrapper(env)

    return env
