
# coding: utf-8
"""
Usage:
    python [options]

Options:
    -h,--help                   显示帮助
    -a,--algorithm=<name>       算法 [default: ppo]
    -c,--copys=<n>              指定并行训练的数量 [default: 1]
    -e,--env=<file>             指定环境名称 [default: None]
    -g,--graphic                是否显示图形界面 [default: False]
    -i,--inference              推断 [default: False]
    -m,--models=<n>             同时训练多少个模型 [default: 1]
    -n,--name=<name>            训练的名字 [default: None]
    -p,--port=<n>               端口 [default: 5005]
    -r,--rnn                    是否使用RNN模型 [default: False]
    -s,--save-frequency=<n>     保存频率 [default: None]
    -t,--train-step=<n>         总的训练次数 [default: None]
    -u,--unity                  是否使用unity客户端 [default: False]
    
    --unity-env=<name>          指定unity环境的名字 [default: None]
    --config-file=<file>        指定模型的超参数config文件 [default: None]
    --store-dir=<file>          指定要保存模型、日志、数据的文件夹路径 [default: None]
    --seed=<n>                  指定模型的随机种子 [default: 0]
    --unity-env-seed=<n>        指定unity环境的随机种子 [default: 0]
    --max-step=<n>              每回合最大步长 [default: None]
    --train-episode=<n>         总的训练回合数 [default: None]
    --train-frame=<n>           总的训练采样次数 [default: None]
    --sampler=<file>            指定随机采样器的文件路径 [default: None]
    --load=<name>               指定载入model的训练名称 [default: None]
    --prefill-steps=<n>         指定预填充的经验数量 [default: None]
    --prefill-choose            指定no_op操作时随机选择动作，或者置0 [default: False]
    --gym                       是否使用gym训练环境 [default: False]
    --gym-env=<name>            指定gym环境的名字 [default: CartPole-v0]
    --gym-env-seed=<n>          指定gym环境的随机种子 [default: 0]
    --render-episode=<n>        指定gym环境从何时开始渲染 [default: None]
    --info=<str>                抒写该训练的描述，用双引号包裹 [default: None]
    --use-wandb                 是否上传数据到W&B [default: False]
Example:
    python run.py -a sac -g -e C:/test.exe -p 6666 -s 10 -n test --config-file config.yaml --max-step 1000 --train-episode 1000 --sampler C:/test_sampler.yaml --unity-env Roller
    python run.py -a ppo -u -n train_in_unity --load last_train_name
    python run.py -ui -a td3 -n inference_in_unity
    python run.py -gi -a dddqn -n inference_with_build -e my_executable_file.exe
    python run.py --gym -a ppo -n train_using_gym --gym-env MountainCar-v0 --render-episode 1000 -c 4
    python run.py -u -a ddpg -n pre_fill --prefill-steps 1000 --prefill-choose
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import sys
sys.path.append('./mlagents')
import time
NAME = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
import platform
BASE_DIR = f'C:/RLData' if platform.system() == "Windows" else os.environ['HOME'] + f'/RLData'

from typing import Dict
from copy import deepcopy
from docopt import docopt
from multiprocessing import Process
from common.agent import Agent
from common.yaml_ops import load_yaml
from common.config import Config


def get_options(options: Dict):
    f = lambda k, t: None if options[k] == 'None' else t(options[k])
    op = Config()
    op.add_dict(dict([
        ['inference',           bool(options['--inference'])],
        ['algo',                str(options['--algorithm'])],
        ['use_rnn',             bool(options['--rnn'])],
        ['algo_config',         f('--config-file', str)],
        ['env',                 f('--env', str)],
        ['port',                int(options['--port'])],
        ['unity',               bool(options['--unity'])],
        ['graphic',             bool(options['--graphic'])],
        ['name',                f('--name', str)],
        ['save_frequency',      f('--save-frequency', int)],
        ['models',              int(options['--models'])],
        ['store_dir',           f('--store-dir', str)],
        ['seed',                int(options['--seed'])],
        ['unity_env_seed',      int(options['--unity-env-seed'])],
        ['max_step_per_episode',f('--max-step', int)],
        ['max_train_step',      f('--train-step', int)],
        ['max_train_frame',     f('--train-frame', int)],
        ['max_train_episode',   f('--train-episode', int)],
        ['sampler',             f('--sampler', str)],
        ['load',                f('--load', str)],
        ['prefill_steps',       f('--prefill-steps', int)],
        ['prefill_choose',      bool(options['--prefill-choose'])],
        ['gym',                 bool(options['--gym'])],
        ['n_copys',             int(options['--copys'])],
        ['gym_env',             str(options['--gym-env'])],
        ['gym_env_seed',        int(options['--gym-env-seed'])],
        ['render_episode',      f('--render-episode', int)],
        ['info',                f('--info', str)],
        ['use_wandb',           bool(options['--use-wandb'])],
        ['unity_env',           f('--unity-env', str)]
    ]))
    return op


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
    options = get_options(dict(options))
    print(options)

    default_config = load_yaml(f'config.yaml')
    # gym > unity > unity_env
    model_args = Config(**default_config['model'])
    train_args = Config(**default_config['train'])
    env_args = Config()
    buffer_args = Config(**default_config['buffer'])

    model_args.algo = options.algo
    model_args.use_rnn = options.use_rnn
    model_args.algo_config = options.algo_config
    model_args.seed = options.seed
    model_args.load = options.load

    env_args.env_num = options.n_copys
    if options.gym:
        train_args.add_dict(default_config['gym']['train'])
        train_args.update({'render_episode': options.render_episode})
        env_args.add_dict(default_config['gym']['env'])
        env_args.type = 'gym'
        env_args.env_name = options.gym_env
        env_args.env_seed = options.gym_env_seed
    else:
        train_args.add_dict(default_config['unity']['train'])
        env_args.add_dict(default_config['unity']['env'])
        env_args.type = 'unity'
        env_args.port = options.port
        env_args.sampler_path = options.sampler
        env_args.env_seed = options.unity_env_seed
        if options.unity:
            env_args.file_path = None
            env_args.env_name = 'unity'
        else:
            env_args.update({'file_path': options.env})
            if os.path.exists(env_args.file_path):
                env_args.env_name = options.unity_env or os.path.join(
                    *os.path.split(env_args.file_path)[0].replace('\\', '/').replace(r'//', r'/').split('/')[-2:]
                )
                if 'visual' in env_args.env_name.lower():
                    # if traing with visual input but do not render the environment, all 0 obs will be passed.
                    options.graphic = True
            else:
                raise Exception('can not find this file.')
        if options.inference:
            env_args.train_mode = False
            env_args.render = True
        else:
            env_args.train_mode = True
            env_args.render = options.graphic

    train_args.index = 0
    train_args.name = NAME
    train_args.use_wandb = options.use_wandb
    train_args.inference = options.inference
    train_args.prefill_choose = options.prefill_choose
    train_args.base_dir = os.path.join(options.store_dir or BASE_DIR, env_args.env_name, model_args.algo)
    train_args.update(
        dict([
            ['name', options.name],
            ['max_step_per_episode', options.max_step_per_episode],
            ['max_train_step', options.max_train_step],
            ['max_train_frame', options.max_train_frame],
            ['max_train_episode', options.max_train_episode],
            ['save_frequency', options.save_frequency],
            ['pre_fill_steps', options.prefill_steps],
            ['info', options.info]
        ])
    )

    if options.inference:
        Agent(env_args, model_args, buffer_args, train_args).evaluate()
        return

    trails = options.models
    if trails == 1:
        agent_run(env_args, model_args, buffer_args, train_args)
    elif trails > 1:
        processes = []
        for i in range(trails):
            _env_args = deepcopy(env_args)
            _model_args = deepcopy(model_args)
            _model_args.seed += i * 10
            _buffer_args = deepcopy(buffer_args)
            _train_args = deepcopy(train_args)
            _train_args.index = i
            if _env_args.type == 'unity':
                _env_args.port = env_args.port + i
            p = Process(target=agent_run, args=(_env_args, _model_args, _buffer_args, _train_args))
            p.start()
            time.sleep(10)
            processes.append(p)
        [p.join() for p in processes]
    else:
        raise Exception('trials must be greater than 0.')


if __name__ == "__main__":
    try:
        import colored_traceback
        colored_traceback.add_hook()
    except ImportError:
        pass
    try:
        run()
    except Exception as e:
        print(e)
        sys.exit()
