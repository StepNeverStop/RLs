from copy import deepcopy
from gym_wrapper import gym_envs
from common.unity_wrapper import UnityWrapper
from common.unity_wrapper import InfoWrapper, UnityReturnWrapper, SamplerWrapper, ActionWrapper


def make_env(env_args):
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
    env = InfoWrapper(env)
    env = UnityReturnWrapper(env)
    env = SamplerWrapper(env, env_args)
    env = ActionWrapper(env)
    return env
