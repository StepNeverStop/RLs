from gym_wrapper import gym_envs
from mlagents.envs import UnityEnvironment
from common.unity_wrapper import UnityWrapper


def make_env(env_args):
    if env_args['type'] == 'gym':
        env = make_gym_env(env_args)
    elif env_args['type'] == 'unity':
        env = make_unity_env(env_args)
    else:
        raise Exception('Unknown environment type.')
    return env


def make_gym_env(env_args):
    env_kargs = {
        'skip': env_args['action_skip'],
        'stack': env_args['obs_stack'],
        'grayscale': env_args['obs_grayscale'],
        'resize': env_args['obs_resize'],
        'scale': env_args['obs_scale'],
    }
    env = gym_envs(gym_env_name=env_args['env_name'],
                   n=env_args['env_num'],
                   seed=env_args['env_seed'],
                   render_mode=env_args['render_mode'],
                   **env_kargs)
    return env


def make_unity_env(env_args):
    if env_args['file_path'] is None:
        env = UnityEnvironment()
    else:
        env = UnityEnvironment(
            file_name=env_args['file_path'],
            base_port=env_args['port'],
            no_graphics=not env_args['render']
        )
    env = UnityWrapper(env, env_args)
    return env
