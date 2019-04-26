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
Example:
    python run.py -a sac -g -e C:/test.exe -p 6666 -s 10 -m test
"""
import os
import sys
import _thread
import Algorithms
from docopt import docopt
from config import train_config
from utils.sth import sth
from mlagents.envs import UnityEnvironment
if sys.platform.startswith('win'):
    import win32api
    import win32con


def run():
    if sys.platform.startswith('win'):
        # Add the _win_handler function to the windows console's handler function list
        win32api.SetConsoleCtrlHandler(_win_handler, 1)

    options = docopt(__doc__)
    print(options)
    reset_config = train_config['reset_config']
    save_frequency = train_config['save_frequency'] if options['--save-frequency'] == 'None' else int(
        options['--save-frequency'])
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
            env_dir = os.path.split(
                options['--env'])[0]
            env_name = env_dir.split('UnityBuild')[-1]
            sys.path.append(env_dir)
            if os.path.exists(env_dir + '/env_config.py'):
                import env_config
                reset_config = env_config.reset_config
            if os.path.exists(env_dir + '/env_loop.py'):
                from env_loop import Loop
        else:
            raise Exception('can not find this file.')
    else:
        env = UnityEnvironment()
        env_name = '/unity'

    if options['--algorithm'] == 'ppo':
        algorithm_config = Algorithms.ppo_config
        model = Algorithms.PPO
        policy_mode = 'ON'
    elif options['--algorithm'] == 'ddpg':
        algorithm_config = Algorithms.ddpg_config
        model = Algorithms.DDPG
        policy_mode = 'OFF'
    elif options['--algorithm'] == 'td3':
        algorithm_config = Algorithms.td3_config
        model = Algorithms.TD3
        policy_mode = 'OFF'
    elif options['--algorithm'] == 'sac':
        algorithm_config = Algorithms.sac_config
        model = Algorithms.SAC
        policy_mode = 'OFF'
    elif options['--algorithm'] == 'sac_no_v':
        algorithm_config = Algorithms.sac_no_v_config
        model = Algorithms.SAC_NO_V
        policy_mode = 'OFF'
    else:
        raise Exception("Don't have this algorithm.")

    if options['--config-file'] != 'None':
        _algorithm_config = sth.load_config(options['--config-file'])
        try:
            for key in _algorithm_config:
                algorithm_config[key]=_algorithm_config[key]
        except Exception as e:
            print(e)
            sys.exit()

    if 'Loop' not in locals().keys():
        from loop import Loop
    base_dir = train_config['base_dir'] + env_name + '/' + \
        options['--algorithm'] + '/' + name + '/'

    print(algorithm_config)
    brain_names = env.external_brain_names
    brains = env.brains
    models = [model(
        s_dim=brains[i].vector_observation_space_size*brains[i].num_stacked_vector_observations,
        a_counts=brains[i].vector_action_space_size[0],
        cp_dir=base_dir + f'{i}' + '/model/',
        log_dir=base_dir + f'{i}' + '/log/',
        excel_dir=base_dir + f'{i}' + '/excel/',
        logger2file=False,
        out_graph=False,
        **algorithm_config
    ) for i in brain_names]

    begin_episode = models[0].get_init_step(
        cp_dir=base_dir + brain_names[0] + '/model/')

    if options['--inference']:
        Loop.inference(env, brain_names, models, reset_config=reset_config)
    else:
        try:
            if policy_mode == 'ON':
                Loop.train_OnPolicy(env, brain_names, models,
                                    begin_episode, save_frequency=save_frequency, reset_config=reset_config)
            else:
                Loop.train_OffPolicy(env, brain_names, models,
                                     begin_episode, save_frequency=save_frequency, reset_config=reset_config)
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
    finally:
        sys.exit()
