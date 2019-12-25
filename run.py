
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
    -m,--modes=<n>              同时训练多少个模型 [default: 1]
    --store-dir=<file>          指定要保存模型、日志、数据的文件夹路径 [default: None]
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
    --gym-env-seed=<n>          指定gym环境的随机种子 [default: 0]
    --render-episode=<n>        指定gym环境从何时开始渲染 [default: None]
    --info=<str>                抒写该训练的描述，用双引号包裹 [default: None]
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
NAME = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
import platform
BASE_DIR = f'C:/RLData' if platform.system() == "Windows" else os.environ['HOME'] + f'/RLData'

from copy import deepcopy
from docopt import docopt
from multiprocessing import Process
from common.agent import Agent
from common.yaml_ops import load_yaml


def agent_run(*args):
    Agent(*args)()


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

    default_config = load_yaml(f'config.yaml')
    # gym > unity > unity_env
    env_args, model_args, train_args = {}, {}, {}
    unity_args, gym_args, buffer_args = default_config['unity'], default_config['gym'], default_config['buffer']

    model_args['logger2file'] = default_config['logger2file']

    model_args['algo'] = str(options['--algorithm'])
    model_args['algo_config'] = None if options['--config-file'] == 'None' else str(options['--config-file'])
    model_args['seed'] = int(options['--seed'])
    model_args['load'] = None if options['--load'] == 'None' else str(options['--load'])

    train_args['index'] = 0
    train_args['all_learner_print'] = default_config['all_learner_print']
    train_args['add_noise2buffer'] = default_config['add_noise2buffer']
    train_args['add_noise2buffer_episode_interval'] = default_config['add_noise2buffer_episode_interval']
    train_args['add_noise2buffer_steps'] = default_config['add_noise2buffer_steps']

    train_args['name'] = NAME if options['--name'] == 'None' else str(options['--name'])
    train_args['max_step'] = default_config['max_step'] if options['--max-step'] == 'None' else int(options['--max-step'])
    train_args['max_episode'] = default_config['max_episode'] if options['--max-episode'] == 'None' else int(options['--max-episode'])
    train_args['save_frequency'] = default_config['save_frequency'] if options['--save-frequency'] == 'None' else int(options['--save-frequency'])
    train_args['inference'] = bool(options['--inference'])
    train_args['fill_in'] = bool(options['--fill-in'])
    train_args['no_op_choose'] = bool(options['--noop-choose'])
    train_args['info'] = default_config['info'] if options['--info'] == 'None' else str(options['--info'])

    if options['--gym']:
        env_args['type'] = 'gym'

        env_args['env_name'] = str(options['--gym-env'])
        env_args['env_num'] = int(options['--gym-agents'])
        env_args['env_seed'] = int(options['--gym-env-seed'])

        env_args['render_mode'] = gym_args['render_mode']
        env_args['action_skip'] = gym_args['action_skip']
        env_args['skip'] = gym_args['skip']
        env_args['obs_stack'] = gym_args['obs_stack']
        env_args['stack'] = gym_args['stack']
        env_args['obs_grayscale'] = gym_args['obs_grayscale']
        env_args['obs_resize'] = gym_args['obs_resize']
        env_args['resize'] = gym_args['resize']
        env_args['obs_scale'] = gym_args['obs_scale']

        train_args['no_op_steps'] = gym_args['random_steps']
        train_args['render'] = gym_args['render']
        train_args['eval_while_train'] = gym_args['eval_while_train']
        train_args['max_eval_episode'] = gym_args['max_eval_episode']
        
        train_args['render_episode'] = gym_args['render_episode'] if options['--render-episode'] == 'None' else int(options['--render-episode'])
    else:
        env_args['type'] = 'unity'
        if options['--unity']:
            env_args['file_path'] = None
            env_args['env_name'] = 'unity'
        else:
            env_args['file_path'] = unity_args['exe_file'] if options['--env'] == 'None' else str(options['--env'])
            if os.path.exists(env_args['file_path']):
                env_args['env_name'] = os.path.join(
                    *os.path.split(env_args['file_path'])[0].replace('\\', '/').replace(r'//', r'/').split('/')[-2:]
                )
            else:
                raise Exception('can not find this file.')
        if bool(options['--inference']):
            env_args['train_mode'] = False
            env_args['render'] = True
        else:
            env_args['train_mode'] = True
            env_args['render'] = bool(options['--graphic'])

        env_args['port'] = int(options['--port'])

        env_args['sampler_path'] = None if options['--sampler'] == 'None' else str(options['--sampler'])
        env_args['reset_config'] = unity_args['reset_config']

        train_args['no_op_steps'] = unity_args['no_op_steps']

    train_args['base_dir'] = os.path.join(
        BASE_DIR if options['--store-dir'] == 'None' else str(options['--store-dir']),
        env_args['env_name'], model_args['algo'])

    if bool(options['--inference']):
        Agent(env_args, model_args, buffer_args, train_args).evaluate()

    trails = int(options['--modes'])
    if trails == 1:
        agent_run(env_args, model_args, buffer_args, train_args)
    elif trails > 1:
        processes = []
        for i in range(trails):
            _env_args = deepcopy(env_args)
            _model_args = deepcopy(model_args)
            _model_args['seed'] += i * 10
            _buffer_args = deepcopy(buffer_args)
            _train_args = deepcopy(train_args)
            _train_args['index'] = i
            if _env_args['type'] == 'unity':
                _env_args['port'] = env_args['port'] + i
            p = Process(target=agent_run, args=(_env_args, _model_args, _buffer_args, _train_args))
            p.start()
            time.sleep(10)
            processes.append(p)
        [p.join() for p in processes]
    else:
        raise Exception('trials must be greater than 0.')


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(e)
        sys.exit()
