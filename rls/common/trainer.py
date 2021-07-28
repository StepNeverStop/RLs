#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
import time
import numpy as np

from typing import (Dict,
                    NoReturn,
                    Optional)

from rls.utils.sundry_utils import set_global_seeds
from rls.utils.time import get_time_hhmmss
from rls.algos import get_model_info
from rls.algos.wrapper.IndependentMA import IndependentMA
from rls.common.train.unity import (unity_train,
                                    unity_no_op,
                                    unity_inference,
                                    ma_unity_no_op,
                                    ma_unity_train,
                                    ma_unity_inference)
from rls.common.train.gym import (gym_train,
                                  gym_no_op,
                                  gym_inference)
from rls.common.yaml_ops import load_config
from rls.common.make_env import make_env
from rls.common.specs import NamedDict
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


class Trainer:
    def __init__(self,
                 env_args: NamedDict,
                 train_args: NamedDict,
                 algo_args: NamedDict):
        '''
        Initilize an agent that consists of training environments, algorithm model.
        params:
            env_args: configurations of training environments
            train_args: configurations of training
        '''
        self.env_args = env_args
        self.train_args = train_args
        self.algo_args = algo_args
        set_global_seeds(self.train_args.seed)

        # ENV
        self.env = make_env(self.env_args)

        # ALGORITHM CONFIG
        self.agent_class, self.policy_mode, self.is_multi = get_model_info(self.train_args.algorithm)

        if self.is_multi:  # TODO: Optimization
            self.algo_args.update({'envspecs': self.env.GroupsSpec})
        else:
            self.algo_args.update({'envspec': self.env.GroupSpec})

        if self.policy_mode == 'on-policy':  # TODO:
            self.train_args.pre_fill_steps = 0  # if on-policy, prefill experience replay is no longer needed.

        self.initialize()
        self.start_time = time.time()

    def initialize(self):
        if not self.is_multi and self.env.is_multi:
            self.model = IndependentMA(self.agent_class, self.algo_args, len(self.env.GroupsSpec))
        else:
            self.model = self.agent_class(**self.algo_args)
        self.model.resume(self.train_args.load_path)
        _train_info = self.model.get_init_training_info()
        self.begin_train_step = _train_info['train_step']
        self.begin_frame_step = _train_info['frame_step']
        self.begin_episode = _train_info['episode']

    def pwi(self, *args, out_time: bool = False) -> NoReturn:
        if self.train_args.allow_print:
            model_info = f'{self.train_args.name} '
            if out_time:
                model_info += f'T: {get_time_hhmmss(self.start_time)} '
            logger.info(''.join([model_info, *args]))

    def __call__(self) -> NoReturn:
        '''
        train
        '''
        if self.train_args.platform == 'gym':
            try:
                gym_no_op(env=self.env,
                          model=self.model,
                          pre_fill_steps=self.train_args.pre_fill_steps,
                          prefill_choose=self.train_args.prefill_choose)
                gym_train(env=self.env,
                          model=self.model,
                          print_func=self.pwi,
                          begin_train_step=self.begin_train_step,
                          begin_frame_step=self.begin_frame_step,
                          begin_episode=self.begin_episode,
                          render=self.train_args.render,
                          render_episode=self.train_args.render_episode,
                          save_frequency=self.train_args.save_frequency,
                          episode_length=self.train_args.episode_length,
                          max_train_episode=self.train_args.max_train_episode,
                          eval_while_train=self.train_args.eval_while_train,
                          max_eval_episode=self.train_args.max_eval_episode,
                          off_policy_step_eval_episodes=self.train_args.off_policy_step_eval_episodes,
                          off_policy_train_interval=self.train_args.off_policy_train_interval,
                          policy_mode=self.policy_mode,
                          moving_average_episode=self.train_args.moving_average_episode,
                          add_noise2buffer=self.train_args.add_noise2buffer,
                          add_noise2buffer_episode_interval=self.train_args.add_noise2buffer_episode_interval,
                          add_noise2buffer_steps=self.train_args.add_noise2buffer_steps,
                          off_policy_eval_interval=self.train_args.off_policy_eval_interval,
                          max_train_step=self.train_args.max_train_step,
                          max_frame_step=self.train_args.max_frame_step)
            finally:
                self.model.close()
                self.env.close()
        else:
            if self.env.is_multi:
                try:
                    ma_unity_no_op(env=self.env,
                                   model=self.model,
                                   pre_fill_steps=self.train_args.pre_fill_steps,
                                   prefill_choose=self.train_args.prefill_choose,
                                   real_done=self.train_args.real_done)
                    ma_unity_train(env=self.env,
                                   model=self.model,
                                   print_func=self.pwi,
                                   begin_train_step=self.begin_train_step,
                                   begin_frame_step=self.begin_frame_step,
                                   begin_episode=self.begin_episode,
                                   save_frequency=self.train_args.save_frequency,
                                   episode_length=self.train_args.episode_length,
                                   max_train_step=self.train_args.max_train_step,
                                   max_frame_step=self.train_args.max_frame_step,
                                   max_train_episode=self.train_args.max_train_episode,
                                   policy_mode=self.policy_mode,
                                   moving_average_episode=self.train_args.moving_average_episode,
                                   real_done=self.train_args.real_done,
                                   off_policy_train_interval=self.train_args.off_policy_train_interval)
                finally:
                    self.model.close()
                    self.env.close()
            else:
                try:
                    unity_no_op(env=self.env,
                                model=self.model,
                                pre_fill_steps=self.train_args.pre_fill_steps,
                                prefill_choose=self.train_args.prefill_choose,
                                real_done=self.train_args.real_done)
                    unity_train(env=self.env,
                                model=self.model,
                                print_func=self.pwi,
                                begin_train_step=self.begin_train_step,
                                begin_frame_step=self.begin_frame_step,
                                begin_episode=self.begin_episode,
                                save_frequency=self.train_args.save_frequency,
                                episode_length=self.train_args.episode_length,
                                max_train_episode=self.train_args.max_train_episode,
                                policy_mode=self.policy_mode,
                                moving_average_episode=self.train_args.moving_average_episode,
                                add_noise2buffer=self.train_args.add_noise2buffer,
                                add_noise2buffer_episode_interval=self.train_args.add_noise2buffer_episode_interval,
                                add_noise2buffer_steps=self.train_args.add_noise2buffer_steps,
                                max_train_step=self.train_args.max_train_step,
                                max_frame_step=self.train_args.max_frame_step,
                                real_done=self.train_args.real_done,
                                off_policy_train_interval=self.train_args.off_policy_train_interval)
                finally:
                    self.model.close()
                    self.env.close()

    def evaluate(self) -> NoReturn:
        if self.train_args.platform == 'gym':
            try:
                gym_inference(env=self.env,
                              model=self.model,
                              episodes=self.train_args.inference_episode)
            finally:
                self.model.close()
                self.env.close()
        else:
            if self.env.is_multi:
                try:
                    ma_unity_inference(env=self.env,
                                       model=self.model,
                                       episodes=self.train_args.inference_episode)
                finally:
                    self.model.close()
                    self.env.close()
            else:
                try:
                    unity_inference(env=self.env,
                                    model=self.model,
                                    episodes=self.train_args.inference_episode)
                finally:
                    self.model.close()
                    self.env.close()

    def apex(self) -> NoReturn:
        if self.policy_mode != 'off-policy':
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
