
#!/usr/bin/env python3
# encoding: utf-8
"""
Usage:
    python [options]

Options:
    -h,--help                   show help info
    -a,--algorithm=<name>       specify the training algorithm [default: ppo]
    -c,--copys=<n>              nums of environment copys that collect data in parallel [default: 1]
    -d, --device=<str>          specify the device that operate Torch.Tensor [default: None]
    -e, --env=<name>            specify the environment name [default: CartPole-v0]
    -f,--file-name=<file>       specify the path of builded training environment of UNITY3D [default: None]
    -g,--graphic                whether show graphic interface when using UNITY3D [default: False]
    -i,--inference              inference the trained model, not train policies [default: False]
    -p,--platform=<str>         specify the platform of training environment [default: gym]
    -l,--load=<name>            specify the name of pre-trained model that need to load [default: None]
    -m,--models=<n>             specify the number of trails that using different random seeds [default: 1]
    -n,--name=<name>            specify the name of this training task [default: None]
    -r,--rnn                    whether use rnn[GRU, LSTM, ...] or not [default: False]
    -s,--save-frequency=<n>     specify the interval that saving model checkpoint [default: None]
    -t,--train-step=<n>         specify the training step that optimize the policy model [default: None]
    -u,--unity                  whether training with UNITY3D editor [default: False]
    --port=<n>                  specify the port that communicate with training environment of UNITY3D [default: 5005]
    --apex=<str>                i.e. "learner"/"worker"/"buffer"/"evaluator" [default: None]
    --config-file=<file>        specify the path of training configuration file [default: None]
    --store-dir=<file>          specify the directory that store model, log and others [default: None]
    --seed=<n>                  specify the random seed of module random, numpy and pytorch [default: 42]
    --env-seed=<n>              specify the environment random seed [default: 42]
    --max-step=<n>              specify the maximum step per episode [default: None]
    --train-episode=<n>         specify the training maximum episode [default: None]
    --train-frame=<n>           specify the training maximum steps interacting with environment [default: None]
    --prefill-steps=<n>         specify the number of experiences that should be collected before start training, use for off-policy algorithms [default: None]
    --prefill-choose            whether choose action using model or choose randomly [default: False]
    --render-episode=<n>        specify when to render the graphic interface of gym environment [default: None]
    --info=<str>                write another information that describe this training task [default: None]
    --hostname                  whether concatenate hostname with the training name [default: False]
    --no-save                   specify whether save models/logs/summaries while training or not [default: False]
Example:
    python run.py
    python run.py -p gym -a dqn -e CartPole-v0 -c 12 -n dqn_cartpole --no-save
    python run.py -p unity -a ppo -n run_with_unity
    python run.py -p unity --file-name /root/env/3dball.app -a sac -n run_with_execution_file
"""

import os
import sys
import time
import logging
import platform

from typing import Dict
from copy import deepcopy
from docopt import docopt
from multiprocessing import Process

from rls.common.trainer import Trainer
from rls.common.config import Config
from rls.common.yaml_ops import load_yaml
from rls.utils.display import show_dict
from rls.utils.logging_utils import set_log_level

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

BASE_DIR = f'C:\RLData' if platform.system() == "Windows" else os.environ['HOME'] + f'/RLData'

set_log_level(logging.INFO)
if sys.platform.startswith('win'):
    import pywintypes   # necessary when using python 3.8+
    import win32api
    import win32con
    import _thread

    def _win_handler(event, hook_sigint=_thread.interrupt_main):
        '''
        handle the event of 'Ctrl+c' in windows operating system.
        '''
        if event == 0:
            hook_sigint()
            return 1
        return 0
    # Add the _win_handler function to the windows console's handler function list
    win32api.SetConsoleCtrlHandler(_win_handler, 1)


def get_options(options: Dict) -> Config:
    '''
    Resolves command-line arguments
    params:
        options: dictionary of command-line arguments
    return:
        op: an instance of Config class that contains the parameters
    '''
    def f(k, t):
        return None if options[k] == 'None' else t(options[k])
    op = Config()
    op.add_dict(dict([
        ['inference',               bool(options['--inference'])],
        ['algo',                    str(options['--algorithm'])],
        ['use_rnn',                 bool(options['--rnn'])],
        ['algo_config',             f('--config-file', str)],
        ['file_name',               f('--file-name', str)],
        ['port',                    int(options['--port'])],
        ['platform',                str(options['--platform'])],
        ['graphic',                 bool(options['--graphic'])],
        ['name',                    f('--name', str)],
        ['device',                  f('--device', str)],
        ['save_frequency',          f('--save-frequency', int)],
        ['models',                  int(options['--models'])],
        ['store_dir',               f('--store-dir', str)],
        ['seed',                    int(options['--seed'])],
        ['env_seed',                int(options['--env-seed'])],
        ['max_step_per_episode',    f('--max-step', int)],
        ['max_train_step',          f('--train-step', int)],
        ['max_train_frame',         f('--train-frame', int)],
        ['max_train_episode',       f('--train-episode', int)],
        ['load',                    f('--load', str)],
        ['prefill_steps',           f('--prefill-steps', int)],
        ['prefill_choose',          bool(options['--prefill-choose'])],
        ['n_copys',                 int(options['--copys'])],
        ['env',                     str(options['--env'])],
        ['render_episode',          f('--render-episode', int)],
        ['info',                    f('--info', str)],
        ['apex',                    f('--apex', str)],
        ['hostname',                bool(options['--hostname'])],
        ['no_save',                 bool(options['--no-save'])],
    ]))
    return op


def agent_run(*args):
    '''
    Start a training task
    '''
    Trainer(*args)()


def main():
    options = docopt(__doc__)
    options = get_options(dict(options))
    show_dict(options.to_dict)

    trails = options.models
    assert trails > 0, '--models must greater than 0.'

    default_config = load_yaml(f'./config.yaml')

    # gym > unity > unity_env
    env_args = Config()
    env_args.env_num = options.n_copys  # Environmental copies of vectorized training.
    env_args.inference = options.inference

    env_args.type = options.platform
    env_args.add_dict(default_config[env_args.type]['env'])
    env_args.env_seed = options.env_seed
    if env_args.type == 'gym':
        env_args.env_name = options.env
    elif env_args.type == 'unity':
        if env_args.initialize_config.env_copys <= 1:
            env_args.initialize_config.env_copys = options.n_copys
        env_args.port = options.port
        env_args.env_seed = options.env_seed
        env_args.render = options.graphic or options.inference

        env_args.file_name = options.file_name
        if env_args.file_name:
            if os.path.exists(env_args.file_name):
                env_args.env_name = options.env or os.path.join(
                    *os.path.split(env_args.file_name)[0].replace('\\', '/').replace(r'//', r'/').split('/')[-2:]
                )
                if 'visual' in env_args.env_name.lower():
                    # if traing with visual input but do not render the environment, all 0 obs will be passed.
                    env_args.render = True
            else:
                raise Exception('can not find the executable file.')
        else:
            env_args.env_name = 'unity'
    else:
        raise NotImplementedError(f'Cannot recognize this kind of platform: {options.platform}')

    train_args = Config(**default_config['train'])
    train_args.add_dict(default_config[env_args.type]['train'])
    if env_args.type == 'gym':
        train_args.render_episode = abs(train_args.render_episode) or sys.maxsize
        train_args.update({'render_episode': options.render_episode})
    train_args.index = 0
    train_args.name = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    train_args.max_step_per_episode = abs(train_args.max_step_per_episode) or sys.maxsize
    train_args.max_train_step = abs(train_args.max_train_step) or sys.maxsize
    train_args.max_frame_step = abs(train_args.max_frame_step) or sys.maxsize
    train_args.max_train_episode = abs(train_args.max_train_episode) or sys.maxsize
    train_args.inference_episode = abs(train_args.inference_episode) or sys.maxsize

    train_args.algo = options.algo
    train_args.device = options.device
    train_args.apex = options.apex
    train_args.use_rnn = options.use_rnn
    train_args.algo_config = options.algo_config
    train_args.seed = options.seed
    train_args.inference = options.inference
    train_args.prefill_choose = options.prefill_choose
    train_args.load_model_path = options.load
    train_args.no_save = options.no_save
    train_args.base_dir = os.path.join(options.store_dir or BASE_DIR, env_args.env_name, train_args.algo)
    if train_args.load_model_path is not None and not os.path.exists(train_args.load_model_path):   # 如果不是绝对路径，就拼接load的训练相对路径
        train_args.load_model_path = os.path.join(train_args.base_dir, train_args.load_model_path)
    train_args.update(dict([
        ['name', options.name],
        ['max_step_per_episode', options.max_step_per_episode],
        ['max_train_step', options.max_train_step],
        ['max_train_frame', options.max_train_frame],
        ['max_train_episode', options.max_train_episode],
        ['save_frequency', options.save_frequency],
        ['pre_fill_steps', options.prefill_steps],
        ['info', options.info]
    ]))
    if options.apex is not None:
        train_args.name = f'{options.apex}/' + train_args.name
    if options.hostname:
        import socket
        train_args.name += ('-' + str(socket.gethostname()))

    # start training
    if options.inference:
        Trainer(env_args, train_args).evaluate()
        return

    if options.apex is not None:
        train_args.update(load_yaml(f'./rls/distribute/apex/config.yaml'))
        Trainer(env_args, train_args).apex()
    else:
        if trails == 1:
            agent_run(env_args, train_args)
        elif trails > 1:
            processes = []
            for i in range(trails):
                _env_args, _train_args = map(deepcopy, [env_args, train_args])
                _train_args.seed += i * 10
                _train_args.name += f'/{i}'
                _train_args.allow_print = True  # NOTE: set this could block other processes' print function
                if _env_args.type == 'unity':
                    _env_args.worker_id = env_args.worker_id + i
                p = Process(target=agent_run, args=(_env_args, _train_args))
                p.start()
                time.sleep(10)
                processes.append(p)
            [p.join() for p in processes]


if __name__ == "__main__":
    try:
        import colored_traceback
        colored_traceback.add_hook()
    except ImportError:
        pass
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit()
