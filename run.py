
#!/usr/bin/env python3
# encoding: utf-8
import os
import sys
import time
import logging
import platform
import argparse
import torch as t

from typing import Dict
from copy import deepcopy
from multiprocessing import Process

from rls.common.trainer import Trainer
from rls.common.specs import NamedDict
from rls.common.yaml_ops import (save_config,
                                 load_config)
from rls.algorithms.register import registry as algo_registry
from rls.envs import platform_list
from rls.utils.display import show_dict
from rls.utils.logging_utils import (set_log_level,
                                     set_log_file)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
set_log_level(logging.INFO)


def get_args():
    '''
    Resolves command-line arguments
    '''
    parser = argparse.ArgumentParser()
    # train and env
    parser.add_argument('-c', '--copys', type=int, default=1,
                        help='nums of environment copys that collect data in parallel')
    parser.add_argument('--seed', type=int, default=42,
                        help='specify the random seed of module random, numpy and pytorch')
    parser.add_argument('-r', '--render', default=False, action='store_true',
                        help='whether render game interface')
    # train
    parser.add_argument('-p', '--platform', type=str, default='gym', choices=platform_list,
                        help='specify the platform of training environment')
    parser.add_argument('-a', '--algorithm', type=str, default='ppo', choices=algo_registry.algo_specs.keys(),
                        help='specify the training algorithm')
    parser.add_argument('-i', '--inference', default=False, action='store_true',
                        help='inference the trained model, not train policies')
    parser.add_argument('-l', '--load-path', type=str, default=None,
                        help='specify the name of pre-trained model that need to load')
    parser.add_argument('-m', '--models', type=int, default=1,
                        help='specify the number of trails that using different random seeds')
    parser.add_argument('-n', '--name', type=str, default=time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())),
                        help='specify the name of this training task')
    parser.add_argument('-s', '--save-frequency', type=int, default=100,
                        help='specify the interval that saving model checkpoint')
    parser.add_argument('--config-file', type=str, default=None,
                        help='specify the path of training configuration file')
    parser.add_argument('--store-dir', type=str, default='./data',
                        help='specify the directory that store model, log and others')  # TODO
    parser.add_argument('--episode-length', type=int, default=1000,
                        help='specify the maximum step per episode')
    parser.add_argument('--prefill-steps', type=int, default=10000,
                        help='specify the number of experiences that should be collected before start training, use for off-policy algorithms')
    parser.add_argument('--hostname', default=False, action='store_true',
                        help='whether concatenate hostname with the training name')
    parser.add_argument('--info', type=str, default='',
                        help='write another information that describe this training task')
    # env
    parser.add_argument('-e', '--env-name', type=str, default='CartPole-v0',
                        help='specify the environment name')
    parser.add_argument('-f', '--file-name', type=str, default=None,
                        help='specify the path of builded training environment of UNITY3D')
    # algo
    parser.add_argument('--no-save', default=False, action='store_true',
                        help='specify whether save models/logs/summaries while training or not')
    parser.add_argument('-d', '--device', type=str, default="cuda" if t.cuda.is_available() else "cpu",
                        help='specify the device that operate Torch.Tensor')
    parser.add_argument('-t', '--max-train-step', type=int, default=sys.maxsize,
                        help='specify the maximum training steps')
    return parser.parse_args()


def main():
    args = get_args()
    assert args.platform in platform_list, "assert args.platform in platform_list"

    train_args = NamedDict()
    train_args.update(load_config(f'rls/configs/train.yaml'))
    train_args.update(load_config(f'rls/configs/{args.platform}/train.yaml'))

    env_args = NamedDict()
    env_args.update(load_config(f'rls/configs/env.yaml'))
    env_args.update(load_config(f'rls/configs/{args.platform}/env.yaml'))

    algo_args = NamedDict()

    if args.config_file is not None:
        custome_config = load_config(args.config_file)
        train_args.update(custome_config['train'])
        env_args.update(custome_config['environment'])
        algo_args.update(load_config(f'rls/configs/algorithms.yaml')[train_args.algorithm])
        algo_args.update(custome_config['algorithm'])
    else:
        copys = 1 if args.inference else args.copys
        train_args.update(args.__dict__)
        algo_args.update(load_config(f'rls/configs/algorithms.yaml')[args.algorithm])
        algo_args.update(dict(n_copys=copys))
        # env config
        env_args.update(dict(platform=args.platform,
                             env_copys=copys,  # Environmental copies of vectorized training.
                             seed=args.seed,
                             inference=args.inference,
                             env_name=args.env_name))

        if env_args.platform == 'unity':
            env_args.env_name = 'UnityEditor'
            env_args.file_name = args.file_name

            if env_args.inference:
                env_args.engine_config.time_scale = 1   # TODO: test
            if env_args.file_name is not None:
                if os.path.exists(env_args.file_name):
                    env_args.env_name = args.env_name or os.path.join(
                        *os.path.split(env_args.file_name)[0].replace('\\', '/').replace(r'//', r'/').split('/')[-2:])
                else:
                    raise Exception('can not find the executable file.')
            # if traing with visual input but do not render the environment, all 0 obs will be passed.
            env_args.render = args.render or args.inference or ('visual' in env_args.env_name.lower())

        # train config
        if args.hostname:
            import socket
            train_args.name += ('-' + str(socket.gethostname()))
        train_args.base_dir = os.path.join(args.store_dir,
                                           train_args.platform,
                                           env_args.env_name,
                                           train_args.algorithm,
                                           train_args.name)
        if train_args.load_path is not None and not os.path.exists(train_args.load_path):   # 如果不是绝对路径，就拼接load的训练相对路径
            train_args.load_path = os.path.join(train_args.base_dir, train_args.load_path)
        # algo config
        algo_args.update({'no_save': args.no_save,
                          'device': args.device,
                          'max_train_step': args.max_train_step,
                          'base_dir': train_args.base_dir})

    # show and save config
    records_dict = {'train': dict(train_args),
                    'environment': dict(env_args),
                    'algorithm': dict(algo_args)}
    show_dict(records_dict)
    if not train_args.inference and not train_args.no_save:
        save_config(train_args.base_dir, records_dict, 'config.yaml')
    # save log
    if train_args.logger2file:
        set_log_file(log_file=os.path.join(train_args.base_dir, 'log.txt'))

    # start training
    if args.inference:
        Trainer(env_args, train_args, algo_args).evaluate()
    else:
        trails = args.models if args.models > 0 else 1
        if trails == 1:
            agent_run(env_args, train_args, algo_args)
        elif trails > 1:
            processes = []
            for i in range(trails):
                _env_args, _train_args, _algo_args = map(deepcopy, [env_args, train_args, algo_args])
                _train_args.seed += i * 10
                _train_args.name += f'/{_train_args.seed}'
                _train_args.allow_print = True  # NOTE: set this could block other processes' print function
                if args.platform == 'unity':
                    _env_args.worker_id = env_args.worker_id + i
                p = Process(target=agent_run, args=(_env_args, _train_args, _algo_args))
                p.start()
                time.sleep(10)
                processes.append(p)
            [p.join() for p in processes]


def agent_run(*args):
    '''
    Start a training task
    '''
    Trainer(*args)()


if __name__ == "__main__":

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
