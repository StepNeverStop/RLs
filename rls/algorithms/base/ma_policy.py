#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from abc import abstractmethod
from collections import defaultdict
from typing import (List,
                    Dict,
                    Union,
                    Callable,
                    Any,
                    Optional,
                    NoReturn)

from rls.algorithms.base.base import Base
from rls.common.specs import EnvGroupArgs


class MultiAgentPolicy(Base):
    def __init__(self,
                 envspecs: List[EnvGroupArgs],
                 batch_size=128,
                 gamma=0.999,
                 max_train_step=1e18,
                 decay_lr=False,
                 representation_net_params={
                     'use_encoder': False,
                     'use_rnn': False,  # always false, using -r to active RNN
                     'vector_net_params': {
                         'network_type': 'adaptive'  # rls.nn.vector_nets
                     },
                     'visual_net_params': {
                         'visual_feature': 128,
                         'network_type': 'simple'  # rls.nn.visual_nets
                     },
                     'encoder_net_params': {
                         'output_dim': 16
                     },
                     'memory_net_params': {
                         'rnn_units': 16,
                         'network_type': 'lstm'
                     }},
                 **kwargs):
        super().__init__(**kwargs)

        self.envspecs = envspecs
        self.n_copys = envspecs[0].n_copys
        self.n_agents_percopy = len(envspecs)

        self.batch_size = batch_size
        self.gamma = gamma
        self.train_step = 0
        self.max_train_step = max_train_step
        self.delay_lr = decay_lr

        self.representation_net_params = dict(representation_net_params)

        self.writers = [self._create_writer(self.log_dir + f'_{i}') for i in range(self.n_agents_percopy)]

    def reset(self) -> Any:
        pass

    def partial_reset(self, done: Union[List, np.ndarray]) -> Any:
        pass

    def intermediate_variable_reset(self) -> NoReturn:
        '''
        TODO: Annotation
        '''
        self.summaries = {
            'model': {},
            **{
                i: {}
                for i in range(self.n_agents_percopy)
            }
        }

    @abstractmethod
    def choose_actions(self, obs, evaluation: bool = False) -> Any:
        '''
        choose actions while training.
        Input: 
            s: vector observation
            visual_s: visual observation
        Output: 
            actions
        '''
        pass

    def _get_actions(self, obs, is_training: bool = True) -> Any:
        '''
        TODO: Annotation
        '''
        raise NotImplementedError

    def write_summaries(self,
                        global_step: Union[int, t.Tensor],
                        summaries: Dict,
                        writer=None) -> NoReturn:
        '''
        write summaries showing in tensorboard.
        '''
        if 'model' in summaries.keys():
            super().write_summaries(global_step, summaries=summaries.pop('model'), writer=self.writer)
        for i, summary in summaries.items():
            super().write_summaries(global_step, summaries=summary, writer=self.writers[i])
