# coding: utf-8
"""
Usage:
    python [options]

Options:
    -h,--help                   显示帮助
    -i,--inference              推断 [default: False]
    -a,--algorithm=<name>       算法 [default: ppo]
    -c,--config-file=<file>     指定模型的超参数config文件 [default: None]
    -e,--env=<file>             指定环境名称 [default: None]
    -p,--port=<n>               端口 [default: 5005]
    -u,--unity                  是否使用unity客户端 [default: False]
    -g,--graphic                是否显示图形界面 [default: False]
    -n,--name=<name>            训练的名字 [default: None]
    -s,--save-frequency=<n>     保存频率 [default: None]
    --max-step=<n>              每回合最大步长 [default: None]
    --sampler=<file>            指定随机采样器的文件路径 [default: None]
    --load=<name>               指定载入model的训练名称 [default: None]
    --gym                       是否使用gym训练环境 [default: False]
    --gym-agents=<n>            指定并行训练的数量 [default: 1]
    --gym-env=<name>            指定gym环境的名字 [default: CartPole-v0]
    --render-episode=<n>        指定gym环境从何时开始渲染 [default: None]
Example:
    python run.py -a sac -g -e C:/test.exe -p 6666 -s 10 -n test -c config.yaml --max-step 1000 --sampler C:/test_sampler.yaml
    python run.py -a ppo -u -n train_in_unity --load last_train_name
    python run.py -ui -a td3 -n inference_in_unity
    python run.py -gi -a dddqn -n inference_with_build -e my_executable_file.exe
    python run.py --gym -a ppo -n train_using_gym --gym-env MountainCar-v0 --render-episode 1000 --gym-agents 4
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import _thread
import Algorithms
from docopt import docopt
from config import train_config
from utils.replay_buffer import ExperienceReplay
from utils.sth import sth

if sys.platform.startswith('win'):
    import win32api
    import win32con

algos = {
    'pg': [Algorithms.pg_config, Algorithms.PG, 'on-policy', 'perEpisode'],
    'ppo': [Algorithms.ppo_config, Algorithms.PPO, 'on-policy', 'perEpisode'],
    'ac': [Algorithms.ac_config, Algorithms.AC, 'off-policy', 'perStep'],  # could be on-policy, but also doesn't work well.
    'a2c': [Algorithms.a2c_config, Algorithms.A2C, 'on-policy', 'perStep'],
    'dpg': [Algorithms.dpg_config, Algorithms.DPG, 'off-policy', 'perStep'],
    'ddpg': [Algorithms.ddpg_config, Algorithms.DDPG, 'off-policy', 'perStep'],
    'td3': [Algorithms.td3_config, Algorithms.TD3, 'off-policy', 'perStep'],
    'sac': [Algorithms.sac_config, Algorithms.SAC, 'off-policy', 'perStep'],
    'sac_no_v': [Algorithms.sac_no_v_config, Algorithms.SAC_NO_V, 'off-policy', 'perStep'],
    'dqn': [Algorithms.dqn_config, Algorithms.DQN, 'off-policy', 'perStep'],
    'ddqn': [Algorithms.ddqn_config, Algorithms.DDQN, 'off-policy', 'perStep'],
    'dddqn': [Algorithms.dddqn_config, Algorithms.DDDQN, 'off-policy', 'perStep'],
    'madpg': [Algorithms.madpg_config, Algorithms.MADPG, 'off-policy', 'perStep'],
    'maddpg': [Algorithms.maddpg_config, Algorithms.MADDPG, 'off-policy', 'perStep'],
    'matd3': [Algorithms.matd3_config, Algorithms.MATD3, 'off-policy', 'perStep'],
}


def _win_handler(event, hook_sigint=_thread.interrupt_main):
    if event == 0:
        hook_sigint()
        return 1
    return 0


def run():
    if sys.platform.startswith('win'):
        # Add the _win_handler function to the windows console's handler function list
        win32api.SetConsoleCtrlHandler(_win_handler, 1)

    options = docopt(__doc__)
    print(options)

    max_step = int(options['--max-step']) if options['--max-step'] != 'None' else train_config['max_step']
    save_frequency = train_config['save_frequency'] if options['--save-frequency'] == 'None' else int(options['--save-frequency'])
    name = train_config['name'] if options['--name'] == 'None' else options['--name']

    # gym > unity > unity_env
    run_params = {
        'options': options,
        'max_step': max_step,
        'save_frequency': save_frequency,
        'name': name
    }
    gym_run(**run_params) if options['--gym'] else unity_run(**run_params)


def unity_run(options, max_step, save_frequency, name):
    from mlagents.envs import UnityEnvironment
    from utils.sampler import create_sampler_manager
    reset_config = train_config['reset_config']
    if options['--env'] != 'None':
        file_name = options['--env']
    else:
        file_name = train_config['exe_file']

    if options['--unity']:
        file_name = None

    if file_name != None:
        if os.path.exists(file_name):
            env = UnityEnvironment(
                file_name=file_name,
                base_port=int(options['--port']),
                no_graphics=False if options['--inference'] else not options['--graphic']
            )
            env_dir = os.path.split(file_name)[0]
            env_name = os.path.join(*env_dir.replace('\\', '/').replace(r'//', r'/').split('/')[-2:])
            sys.path.append(env_dir)
            if os.path.exists(env_dir + '/env_config.py'):
                import env_config
                reset_config = env_config.reset_config
                max_step = env_config.max_step
            if os.path.exists(env_dir + '/env_loop.py'):
                from env_loop import Loop
        else:
            raise Exception('can not find this file.')
    else:
        env = UnityEnvironment()
        env_name = 'unity'

    try:
        algorithm_config, model, policy_mode, train_mode = algos[options['--algorithm']]
        ma = options['--algorithm'][:2] == 'ma'
    except KeyError:
        raise Exception("Don't have this algorithm.")

    if 'Loop' not in locals().keys():
        if ma:
            from ma_loop import Loop
        else:
            from loop import Loop

    sampler_manager, resampling_interval = create_sampler_manager(
        options['--sampler'], env.reset_parameters
    )

    if options['--config-file'] != 'None':
        algorithm_config = update_config(algorithm_config, options['--config-file'])
    _base_dir = os.path.join(train_config['base_dir'], env_name, options['--algorithm'])
    base_dir = os.path.join(_base_dir, name)
    show_config(algorithm_config)

    brain_names = env.external_brain_names
    brains = env.brains
    brain_num = len(brain_names)

    visual_resolutions = {}
    for i in brain_names:
        if brains[i].number_visual_observations:
            visual_resolutions[f'{i}'] = [
                brains[i].camera_resolutions[0]['height'],
                brains[i].camera_resolutions[0]['width'],
                1 if brains[i].camera_resolutions[0]['blackAndWhite'] else 3
            ]
        else:
            visual_resolutions[f'{i}'] = []

    model_params = [{
        's_dim': brains[i].vector_observation_space_size * brains[i].num_stacked_vector_observations,
        'a_dim_or_list': brains[i].vector_action_space_size,
        'action_type': brains[i].vector_action_space_type,
        'base_dir': os.path.join(base_dir, i),
        'logger2file': train_config['logger2file'],
        'out_graph': train_config['out_graph'],
    } for i in brain_names]

    if ma:
        assert brain_num > 1, 'if using ma* algorithms, number of brains must larger than 1'
        data = ExperienceReplay(train_config['ma_batch_size'], train_config['ma_capacity'])
        extra_params = {'data': data}
        models = [model(
            n=brain_num,
            i=i,
            **model_params[i],
            **algorithm_config
        ) for i in range(brain_num)]
    else:
        extra_params = {}
        models = [model(
            visual_sources=brains[i].number_visual_observations,
            visual_resolution=visual_resolutions[f'{i}'],
            **model_params[index],
            **algorithm_config
        ) for index, i in enumerate(brain_names)]

    [models[index].init_or_restore(os.path.join(_base_dir, name if options['--load'] == 'None' else options['--load'], i)) for index, i in enumerate(brain_names)]
    begin_episode = models[0].get_init_episode()
    max_episode = models[0].get_max_episode()

    params = {
        'env': env,
        'brain_names': brain_names,
        'models': models,
        'begin_episode': begin_episode,
        'save_frequency': save_frequency,
        'reset_config': reset_config,
        'max_step': max_step,
        'max_episode': max_episode,
        'train_mode': train_mode,
        'sampler_manager': sampler_manager,
        'resampling_interval': resampling_interval
    }
    no_op_params = {
        'env': env,
        'brain_names': brain_names,
        'models': models,
        'brains': brains,
        'steps': 30
    }
    params.update(extra_params)
    no_op_params.update(extra_params)

    if options['--inference']:
        Loop.inference(env, brain_names, models, reset_config=reset_config, sampler_manager=sampler_manager, resampling_interval=resampling_interval)
    else:
        try:
            [sth.save_config(os.path.join(base_dir, i, 'config'), algorithm_config) for i in brain_names]
            Loop.no_op(**no_op_params)
            Loop.train(**params)
        except Exception as e:
            print(e)
        finally:
            try:
                [models[i].close() for i in range(len(models))]
            except Exception as e:
                print(e)
            finally:
                env.close()
                sys.exit()


def gym_run(options, max_step, save_frequency, name):
    from gym_loop import Loop
    from gym.spaces import Box, Discrete, Tuple
    from gym_wrapper import gym_envs

    available_type = [Box, Discrete]

    render = train_config['gym_render']
    render_episode = int(options['--render-episode']) if options['--render-episode'] != 'None' else train_config['gym_render_episode']

    try:
        env = gym_envs(options['--gym-env'], int(options['--gym-agents']))
        print('obs: ', env.observation_space)
        print('a: ', env.action_space)
        assert env.observation_space in available_type and env.action_space in available_type, 'action_space and observation_space must be one of available_type'
    except Exception as e:
        print(e)

    try:
        algorithm_config, model, policy_mode, train_mode = algos[options['--algorithm']]
    except KeyError:
        raise Exception("Don't have this algorithm.")

    if options['--config-file'] != 'None':
        algorithm_config = update_config(algorithm_config, options['--config-file'])
    _base_dir = os.path.join(train_config['base_dir'], options['--gym-env'], options['--algorithm'])
    base_dir = os.path.join(_base_dir, name)
    show_config(algorithm_config)

    if type(env.observation_space) == Box:
        if len(env.observation_space.shape) == 1:
            s_dim = env.observation_space.shape[0]
        else:
            s_dim = 0
    else:
        s_dim = env.observation_space.n
    if len(env.observation_space.shape) == 3:
        visual_sources = 1
        visual_resolution = list(env.observation_space.shape)
    else:
        visual_sources = 0
        visual_resolution = []

    if type(env.action_space) == Box:
        assert len(env.action_space.shape) == 1, 'if action space is continuous, the shape length of action must equal to 1'
        a_dim_or_list = env.action_space.shape
        action_type = 'continuous'
    elif type(env.action_space) == Tuple:
        assert all([type(i) == Discrete for i in env.action_space]) == True, 'if action space is Tuple, each item in it must have type Discrete'
        a_dim_or_list = [i.n for i in env.action_space]
        action_type = 'discrete'
    else:
        a_dim_or_list = [env.action_space.n]
        action_type = 'discrete'

    gym_model = model(
        s_dim=s_dim,
        visual_sources=visual_sources,
        visual_resolution=visual_resolution,
        a_dim_or_list=a_dim_or_list,
        action_type=action_type,
        base_dir=base_dir,
        logger2file=train_config['logger2file'],
        out_graph=train_config['out_graph'],
        **algorithm_config
    )
    gym_model.init_or_restore(os.path.join(_base_dir, name if options['--load'] == 'None' else options['--load']))
    begin_episode = gym_model.get_init_episode()
    max_episode = gym_model.get_max_episode()
    params = {
        'env': env,
        'gym_model': gym_model,
        'action_type': action_type,
        'begin_episode': begin_episode,
        'save_frequency': save_frequency,
        'max_step': max_step,
        'max_episode': max_episode,
        'render': render,
        'render_episode': render_episode,
        'train_mode': train_mode
    }
    if options['--inference']:
        Loop.inference(env, gym_model, action_type)
    else:
        sth.save_config(os.path.join(base_dir, 'config'), algorithm_config)
        try:
            Loop.no_op(env, gym_model, action_type, 30)
            Loop.train(**params)
        except Exception as e:
            print(e)
        finally:
            try:
                gym_model.close()
            except Exception as e:
                print(e)
            finally:
                env.close()
                sys.exit()


def update_config(config):
    _config = sth.load_config(options['--config-file'])
    try:
        for key in _config:
            config[key] = _config[key]
    except Exception as e:
        print(e)
        sys.exit()
    return config


def show_config(config):
    for key in config:
        print('-' * 46)
        print('|', str(key).ljust(20), str(config[key]).rjust(20), '|')
    print('-' * 46)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(e)
        sys.exit()
