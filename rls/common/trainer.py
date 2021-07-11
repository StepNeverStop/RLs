#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
import time
import numpy as np

from typing import (Dict,
                    NoReturn,
                    Optional)

from rls.utils.display import show_dict
from rls.utils.sundry_utils import (check_or_create,
                                    set_global_seeds)
from rls.utils.time import get_time_hhmmss
from rls.algos import get_model_info
from rls.common.train.unity import (unity_train,
                                    unity_no_op,
                                    unity_inference,
                                    ma_unity_no_op,
                                    ma_unity_train,
                                    ma_unity_inference)
from rls.common.train.gym import (gym_train,
                                  gym_no_op,
                                  gym_inference)
from rls.common.yaml_ops import (save_config,
                                 load_config)
from rls.common.make_env import make_env
from rls.common.config import Config
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


def UpdateConfig(config: Dict, file_path: str, key_name: str = 'algo') -> Dict:
    '''
    update configurations from a readable file.
    params:
        config: current configurations
        file_path: path of configuration file that needs to be loaded
        key_name: a specified key in configuration file that needs to update current configurations
    return:
        config: updated configurations
    '''
    _config = load_config(file_path)
    key_values = _config[key_name]
    try:
        for key in key_values:
            config[key] = key_values[key]
    except Exception as e:
        logger.info(e)
        sys.exit()
    return config


class Trainer:
    def __init__(self, env_args: Config, train_args: Config):
        '''
        Initilize an agent that consists of training environments, algorithm model.
        params:
            env_args: configurations of training environments
            train_args: configurations of training
        '''
        self.env_args = env_args
        self.train_args = train_args
        set_global_seeds(int(self.train_args.seed))

        self._name = self.train_args['name']
        self.train_args['base_dir'] = os.path.join(self.train_args['base_dir'], self.train_args['name'])  # train_args['base_dir'] DIR/ENV_NAME/ALGORITHM_NAME

        self._allow_print = bool(self.train_args.get('allow_print', False))

        # ENV
        self.env = make_env(self.env_args.to_dict)

        # ALGORITHM CONFIG
        self.MODEL, self.algo_args, self.train_args['policy_mode'], _policy_type = get_model_info(self.train_args['algo'])
        self.multi_agents_training = _policy_type == 'multi'

        if self.train_args['algo_config'] is not None:
            self.algo_args = UpdateConfig(self.algo_args, self.train_args['algo_config'], 'algo')
        self.algo_args['representation_net_params']['use_rnn'] = self.train_args['use_rnn']
        self.algo_args['no_save'] = self.train_args['no_save']
        show_dict(self.algo_args)

        if self.train_args['policy_mode'] == 'on-policy':
            self.train_args['pre_fill_steps'] = 0  # if on-policy, prefill experience replay is no longer needed.

        self.initialize()
        self.start_time = time.time()

    def initialize(self):
        if self.multi_agents_training:  # TODO: Optimization
            self.algo_args.update({'envspecs': self.env.GroupsSpec})
        else:
            self.algo_args.update({'envspec': self.env.GroupSpec})

        self.algo_args.update({
            'max_train_step': self.train_args.max_train_step,
            'base_dir': self.train_args.base_dir
        })
        self.model = self.MODEL(**self.algo_args)
        self.model.resume(self.train_args.load_model_path)

        _train_info = self.model.get_init_training_info()
        self.train_args['begin_train_step'] = _train_info['train_step']
        self.train_args['begin_frame_step'] = _train_info['frame_step']
        self.train_args['begin_episode'] = _train_info['episode']
        if not self.train_args['inference'] and not self.train_args['no_save']:
            if self.multi_agents_training:  # TODO: Optimization
                self.algo_args['envspecs'] = str(self.algo_args['envspecs'])
            else:
                self.algo_args['envspec'] = str(self.algo_args['envspec'])
            records_dict = {
                'env': self.env_args.to_dict,
                'train': self.train_args.to_dict,
                'algo': self.algo_args
            }
            save_config(os.path.join(self.train_args.base_dir, 'config'), records_dict)

    def pwi(self, *args, out_time: bool = False) -> NoReturn:
        if self._allow_print:
            model_info = f'{self._name} '
            if out_time:
                model_info += f'T: {get_time_hhmmss(self.start_time)} '
            logger.info(''.join([model_info, *args]))
        else:
            pass

    def __call__(self) -> NoReturn:
        '''
        train
        '''
        if self.env_args['type'] == 'gym':
            try:
                gym_no_op(
                    env=self.env,
                    model=self.model,
                    pre_fill_steps=int(self.train_args['pre_fill_steps']),
                    prefill_choose=bool(self.train_args['prefill_choose'])
                )
                gym_train(
                    env=self.env,
                    model=self.model,
                    print_func=self.pwi,
                    begin_train_step=int(self.train_args['begin_train_step']),
                    begin_frame_step=int(self.train_args['begin_frame_step']),
                    begin_episode=int(self.train_args['begin_episode']),
                    render=bool(self.train_args['render']),
                    render_episode=int(self.train_args.get('render_episode', sys.maxsize)),
                    save_frequency=int(self.train_args['save_frequency']),
                    max_step_per_episode=int(self.train_args['max_step_per_episode']),
                    max_train_episode=int(self.train_args['max_train_episode']),
                    eval_while_train=bool(self.train_args['eval_while_train']),
                    max_eval_episode=int(self.train_args['max_eval_episode']),
                    off_policy_step_eval_episodes=int(self.train_args['off_policy_step_eval_episodes']),
                    off_policy_train_interval=int(self.train_args['off_policy_train_interval']),
                    policy_mode=str(self.train_args['policy_mode']),
                    moving_average_episode=int(self.train_args['moving_average_episode']),
                    add_noise2buffer=bool(self.train_args['add_noise2buffer']),
                    add_noise2buffer_episode_interval=int(self.train_args['add_noise2buffer_episode_interval']),
                    add_noise2buffer_steps=int(self.train_args['add_noise2buffer_steps']),
                    off_policy_eval_interval=int(self.train_args['off_policy_eval_interval']),
                    max_train_step=int(self.train_args['max_train_step']),
                    max_frame_step=int(self.train_args['max_frame_step'])
                )
            finally:
                self.model.close()
                self.env.close()
        else:
            if self.multi_agents_training:
                try:
                    ma_unity_no_op(
                        env=self.env,
                        model=self.model,
                        pre_fill_steps=int(self.train_args['pre_fill_steps']),
                        prefill_choose=bool(self.train_args['prefill_choose']),
                        real_done=bool(self.train_args['real_done'])
                    )
                    ma_unity_train(
                        env=self.env,
                        model=self.model,
                        print_func=self.pwi,
                        begin_train_step=int(self.train_args['begin_train_step']),
                        begin_frame_step=int(self.train_args['begin_frame_step']),
                        begin_episode=int(self.train_args['begin_episode']),
                        save_frequency=int(self.train_args['save_frequency']),
                        max_step_per_episode=int(self.train_args['max_step_per_episode']),
                        max_train_step=int(self.train_args['max_train_step']),
                        max_frame_step=int(self.train_args['max_frame_step']),
                        max_train_episode=int(self.train_args['max_train_episode']),
                        policy_mode=str(self.train_args['policy_mode']),
                        moving_average_episode=int(self.train_args['moving_average_episode']),
                        real_done=bool(self.train_args['real_done']),
                        off_policy_train_interval=int(self.train_args['off_policy_train_interval'])
                    )
                finally:
                    self.model.close()
                    self.env.close()
            else:
                try:
                    unity_no_op(
                        env=self.env,
                        model=self.model,
                        pre_fill_steps=int(self.train_args['pre_fill_steps']),
                        prefill_choose=bool(self.train_args['prefill_choose']),
                        real_done=bool(self.train_args['real_done'])
                    )
                    unity_train(
                        env=self.env,
                        model=self.model,
                        print_func=self.pwi,
                        begin_train_step=int(self.train_args['begin_train_step']),
                        begin_frame_step=int(self.train_args['begin_frame_step']),
                        begin_episode=int(self.train_args['begin_episode']),
                        save_frequency=int(self.train_args['save_frequency']),
                        max_step_per_episode=int(self.train_args['max_step_per_episode']),
                        max_train_episode=int(self.train_args['max_train_episode']),
                        policy_mode=str(self.train_args['policy_mode']),
                        moving_average_episode=int(self.train_args['moving_average_episode']),
                        add_noise2buffer=bool(self.train_args['add_noise2buffer']),
                        add_noise2buffer_episode_interval=int(self.train_args['add_noise2buffer_episode_interval']),
                        add_noise2buffer_steps=int(self.train_args['add_noise2buffer_steps']),
                        max_train_step=int(self.train_args['max_train_step']),
                        max_frame_step=int(self.train_args['max_frame_step']),
                        real_done=bool(self.train_args['real_done']),
                        off_policy_train_interval=int(self.train_args['off_policy_train_interval'])
                    )
                finally:
                    self.model.close()
                    self.env.close()

    def evaluate(self) -> NoReturn:
        if self.env_args['type'] == 'gym':
            try:
                gym_inference(
                    env=self.env,
                    model=self.model,
                    episodes=self.train_args['inference_episode']
                )
            finally:
                self.model.close()
                self.env.close()
        else:
            if self.multi_agents_training:
                try:
                    ma_unity_inference(
                        env=self.env,
                        model=self.model,
                        episodes=self.train_args['inference_episode']
                    )
                finally:
                    self.model.close()
                    self.env.close()
            else:
                try:
                    unity_inference(
                        env=self.env,
                        model=self.model,
                        episodes=self.train_args['inference_episode']
                    )
                finally:
                    self.model.close()
                    self.env.close()

    def apex(self) -> NoReturn:
        if self.train_args['policy_mode'] != 'off-policy':
            raise Exception('Ape-X only suitable for off-policy algorithms.')

        if self.train_args['apex'] == 'learner':
            from rls.distribute.apex.learner import learner
            learner(
                env=self.env,
                model=self.model,
                ip=self.train_args['apex_learner_ip'],
                port=self.train_args['apex_learner_port']
            )
        elif self.train_args['apex'] == 'worker':
            from rls.distribute.apex.worker import worker
            worker(
                env=self.env,
                model=self.model,
                learner_ip=self.train_args['apex_learner_ip'],
                learner_port=self.train_args['apex_learner_port'],
                buffer_ip=self.train_args['apex_buffer_ip'],
                buffer_port=self.train_args['apex_buffer_port'],
                worker_args=self.train_args['apex_worker_args']
            )
        elif self.train_args['apex'] == 'buffer':
            from rls.distribute.apex.buffer import buffer
            buffer(
                ip=self.train_args['apex_buffer_ip'],
                port=self.train_args['apex_buffer_port'],
                learner_ip=self.train_args['apex_learner_ip'],
                learner_port=self.train_args['apex_learner_port'],
                buffer_args=self.train_args['apex_buffer_args']
            )
        elif self.train_args['apex'] == 'evaluator':
            from rls.distribute.apex.evaluator import evaluator
            evaluator(
                env=self.env,
                model=self.model,
                learner_ip=self.train_args['apex_learner_ip'],
                learner_port=self.train_args['apex_learner_port'],
                evaluator_args=self.train_args['apex_evaluator_args']
            )

        return
