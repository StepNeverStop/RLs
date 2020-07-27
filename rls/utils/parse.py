#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
NAME = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
import platform
BASE_DIR = f'C:\RLData' if platform.system() == "Windows" else os.environ['HOME'] + f'/RLData'

from typing import Dict

from rls.common.config import Config


def parse_options(options: Config, default_config: Dict):
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

    env_args.env_num = options.n_copys  # Environmental copies of vectorized training.
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
                raise Exception('can not find the executable file.')
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
    return env_args, model_args, buffer_args, train_args
