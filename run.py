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
Example:
    python run.py -a sac -g -e C:/test.exe -p 6666 -s 10 -n test -c config.yaml --max-step 1000 --sampler C:/test_sampler.yaml
"""
import os
import sys
import _thread
import Algorithms
from docopt import docopt
from config import train_config
from utils.sth import sth
from mlagents.envs import UnityEnvironment
from utils.sampler import create_sampler_manager
if sys.platform.startswith('win'):
    import win32api
    import win32con

algos = {
    'pg': [Algorithms.pg_config, Algorithms.PG, 'on-policy', 'perEpisode'],
    'ppo': [Algorithms.ppo_config, Algorithms.PPO, 'on-policy', 'perEpisode'],
    'offpg': [Algorithms.offpg_config, Algorithms.OFFPG, 'off-policy', 'perStep'],
    'ac': [Algorithms.ac_config, Algorithms.AC, 'on-policy', 'perStep'],
    'a2c': [Algorithms.a2c_config, Algorithms.A2C, 'on-policy', 'perStep'],
    'ddpg': [Algorithms.ddpg_config, Algorithms.DDPG, 'off-policy', 'perStep'],
    'td3': [Algorithms.td3_config, Algorithms.TD3, 'off-policy', 'perStep'],
    'sac': [Algorithms.sac_config, Algorithms.SAC, 'off-policy', 'perStep'],
    'sac_no_v': [Algorithms.sac_no_v_config, Algorithms.SAC_NO_V, 'off-policy', 'perStep'],
    'std': [Algorithms.std_config, Algorithms.STD, 'off-policy', 'perStep'],
    'dqn': [Algorithms.dqn_config, Algorithms.DQN, 'off-policy', 'perStep'],
    'ddqn': [Algorithms.ddqn_config, Algorithms.DDQN, 'off-policy', 'perStep'],
    'dddqn': [Algorithms.dddqn_config, Algorithms.DDDQN, 'off-policy', 'perStep']
}

def run():
    if sys.platform.startswith('win'):
        # Add the _win_handler function to the windows console's handler function list
        win32api.SetConsoleCtrlHandler(_win_handler, 1)

    options = docopt(__doc__)
    print(options)
    reset_config = train_config['reset_config']
    max_step = train_config['max_step']
    save_frequency = train_config['save_frequency'] if options['--save-frequency'] == 'None' else int(options['--save-frequency'])
    name = train_config['name'] if options['--name'] == 'None' else options['--name']
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

    sampler_manager, resampling_interval = create_sampler_manager(
        options['--sampler'], env.reset_parameters
    )
    
    try:
        algorithm_config, model, policy_mode, train_mode = algos[options['--algorithm']]
    except KeyError:
        raise Exception("Don't have this algorithm.")

    if options['--config-file'] != 'None':
        _algorithm_config = sth.load_config(options['--config-file'])
        try:
            for key in _algorithm_config:
                algorithm_config[key] = _algorithm_config[key]
        except Exception as e:
            print(e)
            sys.exit()

    if 'Loop' not in locals().keys():
        from loop import Loop
    base_dir = os.path.join(train_config['base_dir'], env_name, options['--algorithm'], name)

    for key in algorithm_config:
        print('-' * 46)
        print('|', str(key).ljust(20), str(algorithm_config[key]).rjust(20), '|')
    print('-' * 46)
    
    if options['--max-step'] != 'None':
        max_step = int(options['--max-step'])
    brain_names = env.external_brain_names
    brains = env.brains

    models = [model(
        s_dim=brains[i].vector_observation_space_size * brains[i].num_stacked_vector_observations,
        visual_sources=brains[i].number_visual_observations,
        visual_resolutions=brains[i].camera_resolutions,
        a_dim_or_list=brains[i].vector_action_space_size,
        action_type=brains[i].vector_action_space_type,
        cp_dir=os.path.join(base_dir, i, 'model'),
        log_dir=os.path.join(base_dir, i, 'log'),
        excel_dir=os.path.join(base_dir, i, 'excel'),
        logger2file=False,
        out_graph=True,
        **algorithm_config
    ) for i in brain_names]

    begin_episode = models[0].get_init_step()
    max_episode = models[0].get_max_episode()

    if options['--inference']:
        Loop.inference(env, brain_names, models, reset_config=reset_config, sampler_manager=sampler_manager, resampling_interval=resampling_interval)
    else:
        [sth.save_config(os.path.join(base_dir, i, 'config'), algorithm_config) for i in brain_names]
        try:
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
                'resampling_interval': resampling_interval
            }
            Loop.no_op(env, brain_names, models, brains, 30)
            if train_mode == 'perEpisode':
                Loop.train_perEpisode(**params)
            else:
                Loop.train_perStep(**params)
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


def _win_handler(event, hook_sigint=_thread.interrupt_main):
    if event == 0:
        hook_sigint()
        return 1
    return 0


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(e)
        sys.exit()
