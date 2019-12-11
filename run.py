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
    --seed=<n>                  指定模型的随机种子 [default: 0]
    --max-step=<n>              每回合最大步长 [default: None]
    --max-episode=<n>           总的训练回合数 [default: None]
    --sampler=<file>            指定随机采样器的文件路径 [default: None]
    --load=<name>               指定载入model的训练名称 [default: None]
    --fill-in                   指定是否预填充经验池至batch_size [default: False]
    --noop-choose               指定no_op操作时随机选择动作，或者置0 [default: False]
    --gym                       是否使用gym训练环境 [default: False]
    --gym-agents=<n>            指定并行训练的数量 [default: 1]
    --gym-env=<name>            指定gym环境的名字 [default: CartPole-v0]
    --gym-env-seed=<n>          指定gym环境的随机种子 [default: 10]
    --gym-models=<n>            同时训练多少个模型 [default: 1]
    --render-episode=<n>        指定gym环境从何时开始渲染 [default: None]
Example:
    python run.py -a sac -g -e C:/test.exe -p 6666 -s 10 -n test -c config.yaml --max-step 1000 --max-episode 1000 --sampler C:/test_sampler.yaml
    python run.py -a ppo -u -n train_in_unity --load last_train_name
    python run.py -ui -a td3 -n inference_in_unity
    python run.py -gi -a dddqn -n inference_with_build -e my_executable_file.exe
    python run.py --gym -a ppo -n train_using_gym --gym-env MountainCar-v0 --render-episode 1000 --gym-agents 4
    python run.py -u -a ddpg -n pre_fill--fill-in --noop-choose
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import sys
import time

from docopt import docopt
from multiprocessing import Process

from utils.sth import sth
from config import train_config
from utils.replay_buffer import ExperienceReplay
from Algorithms.register import get_model_info


def run():
    if sys.platform.startswith('win'):
        import win32api
        import win32con
        import _thread

        def _win_handler(event, hook_sigint=_thread.interrupt_main):
            if event == 0:
                hook_sigint()
                return 1
            return 0
        # Add the _win_handler function to the windows console's handler function list
        win32api.SetConsoleCtrlHandler(_win_handler, 1)

    options = docopt(__doc__)
    print(options)

    max_step = train_config['max_step'] if options['--max-step'] == 'None' else int(options['--max-step'])
    max_episode = train_config['max_episode'] if options['--max-episode'] == 'None' else int(options['--max-episode'])
    save_frequency = train_config['save_frequency'] if options['--save-frequency'] == 'None' else int(options['--save-frequency'])
    name = train_config['name'] if options['--name'] == 'None' else options['--name']
    seed = int(options['--seed'])
    share_args, unity_args, gym_args = train_config['share'], train_config['unity'], train_config['gym']

    # gym > unity > unity_env
    run_params = {
        'share_args': share_args,
        'options': options,
        'max_step': max_step,
        'max_episode': max_episode,
        'save_frequency': save_frequency,
        'name': name,
        'seed': seed,
    }
    if options['--gym']:
        trails = int(options['--gym-models'])
        if trails == 1:
            gym_run(default_args=gym_args, **run_params)
        elif trails > 1:
            processes = []
            for i in range(trails):
                p = Process(target=gym_run,
                            args=(
                                gym_args,
                                share_args,
                                options,
                                max_step,
                                max_episode,
                                save_frequency,
                                name + '-' + str(i),
                                seed + i * 10
                            ))
                p.start()
                time.sleep(1)
                processes.append(p)
            [p.join() for p in processes]
        else:
            raise Exception('trials must be greater than 0.')
    else:
        unity_run(default_args=unity_args, **run_params)


def unity_run(default_args, share_args, options, max_step, max_episode, save_frequency, name, seed):
    from mlagents.envs import UnityEnvironment
    from utils.sampler import create_sampler_manager

    model, algorithm_config, policy_mode = get_model_info(options['--algorithm'])
    ma = options['--algorithm'][:3] == 'ma_'

    reset_config = default_args['reset_config']
    if options['--unity']:
        env = UnityEnvironment()
        env_name = 'unity'
    else:
        file_name = default_args['exe_file'] if options['--env'] == 'None' else options['--env']
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
    sampler_manager, resampling_interval = create_sampler_manager(options['--sampler'], env.reset_parameters)

    if 'Loop' not in locals().keys():
        if ma:
            from ma_loop import Loop
        else:
            from loop import Loop

    if options['--config-file'] != 'None':
        algorithm_config = update_config(algorithm_config, options['--config-file'])
    _base_dir = os.path.join(share_args['base_dir'], env_name, options['--algorithm'])
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
        's_dim': brains[b].vector_observation_space_size * brains[b].num_stacked_vector_observations,
        'a_dim_or_list': brains[b].vector_action_space_size,
        'is_continuous': True if brains[b].vector_action_space_type == 'continuous' else False,
        'max_episode': max_episode,
        'base_dir': os.path.join(base_dir, b),
        'logger2file': share_args['logger2file'],
        'seed': seed + i * 10,
    } for i, b in enumerate(brain_names)]

    if ma:
        assert brain_num > 1, 'if using ma* algorithms, number of brains must larger than 1'
        data = ExperienceReplay(share_args['ma']['batch_size'], share_args['ma']['capacity'])
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

    params = {
        'env': env,
        'brain_names': brain_names,
        'models': models,
        'begin_episode': begin_episode,
        'save_frequency': save_frequency,
        'reset_config': reset_config,
        'max_step': max_step,
        'max_episode': max_episode,
        'sampler_manager': sampler_manager,
        'resampling_interval': resampling_interval,
        'policy_mode': policy_mode
    }
    if 'batch_size' in algorithm_config.keys() and options['--fill-in']:
        steps = algorithm_config['batch_size']
    else:
        steps = default_args['no_op_steps']
    no_op_params = {
        'env': env,
        'brain_names': brain_names,
        'models': models,
        'brains': brains,
        'steps': steps,
        'choose': options['--noop-choose']
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
            [models[i].close() for i in range(len(models))]
            env.close()
            sys.exit()


def gym_run(default_args, share_args, options, max_step, max_episode, save_frequency, name, seed):
    from gym_loop import Loop
    from gym_wrapper import gym_envs

    model, algorithm_config, policy_mode = get_model_info(options['--algorithm'])
    render_episode = int(options['--render-episode']) if options['--render-episode'] != 'None' else default_args['render_episode']

    try:
        env = gym_envs(gym_env_name=options['--gym-env'],
                       n=int(options['--gym-agents']),
                       seed=int(options['--gym-env-seed']),
                       render_mode=default_args['render_mode'])
    except Exception as e:
        print(e)

    if options['--config-file'] != 'None':
        algorithm_config = update_config(algorithm_config, options['--config-file'])
    _base_dir = os.path.join(share_args['base_dir'], options['--gym-env'], options['--algorithm'])
    base_dir = os.path.join(_base_dir, name)
    show_config(algorithm_config)

    model_params = {
        's_dim': env.s_dim,
        'visual_sources': env.visual_sources,
        'visual_resolution': env.visual_resolution,
        'a_dim_or_list': env.a_dim_or_list,
        'is_continuous': env.is_continuous,
        'max_episode': max_episode,
        'base_dir': base_dir,
        'logger2file': share_args['logger2file'],
        'seed': seed,
    }
    gym_model = model(
        **model_params,
        **algorithm_config
    )
    gym_model.init_or_restore(os.path.join(_base_dir, name if options['--load'] == 'None' else options['--load']))
    begin_episode = gym_model.get_init_episode()
    params = {
        'env': env,
        'gym_model': gym_model,
        'begin_episode': begin_episode,
        'save_frequency': save_frequency,
        'max_step': max_step,
        'max_episode': max_episode,
        'eval_while_train': default_args['eval_while_train'],  # whether to eval while training.
        'max_eval_episode': default_args['max_eval_episode'],
        'render': default_args['render'],
        'render_episode': render_episode,
        'policy_mode': policy_mode
    }
    if 'batch_size' in algorithm_config.keys() and options['--fill-in']:
        steps = algorithm_config['batch_size']
    else:
        steps = default_args['random_steps']
    if options['--inference']:
        Loop.inference(env, gym_model)
    else:
        sth.save_config(os.path.join(base_dir, 'config'), algorithm_config)
        try:
            Loop.no_op(env, gym_model, steps, choose=options['--noop-choose'])
            Loop.train(**params)
        except Exception as e:
            print(e)
        finally:
            gym_model.close()
            env.close()
            sys.exit()


def update_config(config, file_path):
    _config = sth.load_config(file_path)
    try:
        for key in _config:
            config[key] = _config[key]
    except Exception as e:
        print(e)
        sys.exit()
    return config


def show_config(config):
    for key in config:
        print('-' * 60)
        print('|', str(key).ljust(28), str(config[key]).rjust(28), '|')
    print('-' * 60)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(e)
        sys.exit()
