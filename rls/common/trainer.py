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
from rls.algorithms import get_model_info
from rls.algorithms.wrapper.IndependentMA import IndependentMA
from rls.train.train import (train,
                             prefill,
                             inference)
from rls.common.yaml_ops import load_config
from rls.envs.make_env import make_env
from easydict import EasyDict
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


class Trainer:
    def __init__(self,
                 env_args: EasyDict,
                 train_args: EasyDict,
                 algo_args: EasyDict):
        '''
        Initilize an agent that consists of training environments, algorithm agent.
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
        # logger.info(self.env.AgentSpecs)

        # ALGORITHM CONFIG
        self.agent_class, self.is_multi = get_model_info(
            self.train_args.algorithm)
        if self.agent_class.policy_mode == 'on-policy':
            self.algo_args.buffer_size = self.train_args.episode_length * self.algo_args.n_copys

        logger.info('Initialize Agent Begin.')
        if not self.is_multi:
            self.agent = IndependentMA(self.agent_class,
                                       self.env.AgentSpecs,
                                       self.algo_args)
        else:
            self.agent = self.agent_class(agent_specs=self.env.AgentSpecs,
                                          state_spec=self.env.StateSpec,
                                          **self.algo_args)
        logger.info('Initialize Agent Successfully.')
        self.agent.resume(self.train_args.load_path)

        self.start_time = time.time()

    def print_function(self, *args, out_time: bool = False) -> NoReturn:
        if self.train_args.allow_print:
            model_info = f'{self.train_args.name} '
            if out_time:
                model_info += f'T: {get_time_hhmmss(self.start_time)} '
            logger.info(''.join([model_info, *args]))

    def __call__(self) -> NoReturn:
        '''
        train
        '''
        try:
            prefill(env=self.env,
                    agent=self.agent,
                    reset_config=self.train_args.reset_config,
                    step_config=self.train_args.step_config,
                    prefill_steps=self.train_args.prefill_steps)
            train(env=self.env,
                  agent=self.agent,
                  print_func=self.print_function,
                  episode_length=self.train_args.episode_length,
                  moving_average_episode=self.train_args.moving_average_episode,
                  render=self.train_args.render,
                  reset_config=self.train_args.reset_config,
                  step_config=self.train_args.step_config)
        finally:
            self.agent.close()
            self.env.close()

    def evaluate(self) -> NoReturn:
        try:
            inference(env=self.env,
                      agent=self.agent,
                      print_func=self.print_function,
                      moving_average_episode=self.train_args.moving_average_episode,
                      reset_config=self.train_args.reset_config,
                      step_config=self.train_args.step_config,
                      episodes=self.train_args.inference_episode)
        finally:
            self.agent.close()
            self.env.close()
