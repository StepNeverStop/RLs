#!/usr/bin/env python3
# encoding: utf-8

from copy import deepcopy

from easydict import EasyDict

from rls.utils.logging_utils import get_logger

logger = get_logger(__name__)


def make_env(env_kargs: EasyDict):
    logger.info('Initialize environment begin...')

    copied_env_kargs = deepcopy(env_kargs)

    if env_kargs.platform == 'gym':
        from rls.envs.gym.env import GymEnv
        env = GymEnv(**copied_env_kargs)
    elif env_kargs.platform == 'unity':
        from rls.envs.unity.env import UnityEnv
        env = UnityEnv(**copied_env_kargs)
    elif env_kargs.platform == 'pettingzoo':
        from rls.envs.pettingzoo.env import PettingZooEnv
        env = PettingZooEnv(**copied_env_kargs)
    else:
        raise Exception('Unknown environment type.')

    logger.info('Initialize environment successful.')
    return env
