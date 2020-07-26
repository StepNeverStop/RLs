#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from copy import deepcopy
from typing import Dict
from envs.wrappers.gym_wrapper import gym_envs
from envs.wrappers.unity_wrapper import UnityWrapper, InfoWrapper, UnityReturnWrapper, SamplerWrapper, ActionWrapper, StackVisualWrapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("common.make_env")


def make_env(env_args: Dict):
    if env_args['type'] == 'gym':
        env = make_gym_env(env_args)
    elif env_args['type'] == 'unity':
        env = make_unity_env(env_args)
    else:
        raise Exception('Unknown environment type.')
    return env


def make_gym_env(env_args):
    env_kargs = deepcopy(env_args)
    env = gym_envs(env_kargs)
    return env


def make_unity_env(env_args):
    env_kargs = deepcopy(env_args)
    env = UnityWrapper(env_kargs)
    logger.debug('Unity UnityWrapper success.')

    env = InfoWrapper(env, env_args)
    logger.debug('Unity InfoWrapper success.')

    if env_args['stack_visual_nums'] > 1:
        env = StackVisualWrapper(env, stack_nums=env_args['stack_visual_nums'])
        logger.debug('Unity StackVisualWrapper success.')
    else:
        env = UnityReturnWrapper(env)
        logger.debug('Unity UnityReturnWrapper success.')

    env = SamplerWrapper(env, env_args)
    logger.debug('Unity SamplerWrapper success.')

    env = ActionWrapper(env)
    logger.debug('Unity ActionWrapper success.')

    return env
