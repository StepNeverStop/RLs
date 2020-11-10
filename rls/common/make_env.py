#!/usr/bin/env python3
# encoding: utf-8

from typing import Dict
from copy import deepcopy

from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


def make_env(env_args: Dict):
    logger.info('Initialize environment begin...')
    if env_args['type'] == 'gym':
        env = make_gym_env(env_args)
    elif env_args['type'] == 'unity':
        env = make_unity_env(env_args)
    else:
        raise Exception('Unknown environment type.')
    logger.info('Initialize environment successful.')
    return env


def make_gym_env(env_args: Dict):
    from rls.envs.gym_wrapper import gym_envs

    env_kargs = deepcopy(env_args)
    env = gym_envs(env_kargs)
    return env


def make_unity_env(env_args: Dict):
    from rls.envs.unity_wrapper import (UnityWrapper,
                                        InfoWrapper,
                                        UnityReturnWrapper,
                                        GrayVisualWrapper,
                                        ResizeVisualWrapper,
                                        ScaleVisualWrapper,
                                        ActionWrapper,
                                        StackVisualWrapper)

    env_kargs = deepcopy(env_args)
    env = UnityWrapper(env_kargs)
    logger.debug('Unity UnityWrapper success.')

    env = InfoWrapper(env, env_args)
    logger.debug('Unity InfoWrapper success.')

    if env_kargs['obs_grayscale']:
        env = GrayVisualWrapper(env)

    if env_kargs['obs_resize']:
        env = ResizeVisualWrapper(env, resize=env_kargs['resize'])

    if env_kargs['obs_scale']:
        env = ScaleVisualWrapper(env)

    env = UnityReturnWrapper(env)

    if env_kargs['obs_stack']:
        env = StackVisualWrapper(env, stack_nums=env_args['stack_visual_nums'])

    env = ActionWrapper(env)
    logger.debug('Unity ActionWrapper success.')

    env.initialize()

    return env
