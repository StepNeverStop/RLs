from copy import deepcopy
from gym_wrapper import gym_envs
from mlagents.mlagents_envs.environment import UnityEnvironment
from mlagents.mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from common.unity_wrapper import InfoWrapper, UnityReturnWrapper, SamplerWrapper


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
    engine_configuration_channel = EngineConfigurationChannel()
    engine_configuration_channel.set_configuration_parameters(time_scale=100)
    # engine_configuration_channel.set_configuration_parameters(width=1028, height=720, quality_level=5, time_scale=0, target_frame_rate=60)
    if env_args['file_path'] is None:
        env = UnityEnvironment(base_port=5004, 
                               seed=env_args['env_seed'],
                               side_channels = [engine_configuration_channel])
    else:
        env = UnityEnvironment(
            file_name=env_args['file_path'],
            base_port=env_args['port'],
            no_graphics=not env_args['render'],
            seed=env_args['env_seed']
        )
    env = InfoWrapper(env)
    env = UnityReturnWrapper(env)
    # env = SamplerWrapper(env, env_args)
    return env
