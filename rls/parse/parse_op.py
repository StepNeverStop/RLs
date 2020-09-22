#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import platform
BASE_DIR = f'C:\RLData' if platform.system() == "Windows" else os.environ['HOME'] + f'/RLData'

from typing import \
    Dict, \
    Tuple

from rls.common.config import Config


def parse_options(options: Config, default_config: Dict) -> Tuple[Config]:
    # gym > unity > unity_env
    env_args = Config()
    env_args.env_num = options.n_copys  # Environmental copies of vectorized training.
    env_args.inference = options.inference  # Environmental copies of vectorized training.
    if options.gym:
        env_args.type = 'gym'
        env_args.add_dict(default_config['gym']['env'])
        env_args.env_name = options.gym_env
        env_args.env_seed = options.gym_env_seed
    else:
        env_args.type = 'unity'
        env_args.add_dict(default_config['unity']['env'])
        env_args.port = options.port
        env_args.env_seed = options.unity_env_seed

        if options.inference:
            env_args.train_mode = False
            env_args.render = True
        else:
            env_args.train_mode = True
            env_args.render = options.graphic

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
                    env_args.render = True
            else:
                raise Exception('can not find the executable file.')

    train_args = Config(**default_config['train'])
    if options.gym:
        train_args.add_dict(default_config['gym']['train'])
        train_args.update({'render_episode': options.render_episode})
    else:
        train_args.add_dict(default_config['unity']['train'])
    train_args.index = 0
    train_args.name = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    train_args.max_train_step = abs(train_args.max_train_step) or sys.maxsize
    train_args.max_frame_step = abs(train_args.max_frame_step) or sys.maxsize
    train_args.max_train_episode = abs(train_args.max_train_episode) or sys.maxsize
    train_args.inference_episode = abs(train_args.inference_episode) or sys.maxsize

    train_args.algo = options.algo
    train_args.apex = options.apex
    train_args.use_rnn = options.use_rnn
    train_args.algo_config = options.algo_config
    train_args.seed = options.seed
    train_args.use_wandb = options.use_wandb
    train_args.inference = options.inference
    train_args.prefill_choose = options.prefill_choose
    train_args.load_model_path = options.load
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

    buffer_args = Config(**default_config['buffer'])
    return env_args, buffer_args, train_args
