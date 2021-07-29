#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from abc import abstractmethod
from collections import defaultdict
from typing import (Union,
                    List,
                    Callable,
                    Tuple,
                    Any,
                    Dict,
                    NoReturn,
                    Optional)

from rls.algorithms.base.base import Base
from rls.utils.vector_runing_average import (DefaultRunningAverage,
                                             SimpleRunningAverage)
from rls.common.specs import EnvGroupArgs
from rls.nn.represent_nets import DefaultRepresentationNetwork
from rls.nn.modules import CuriosityModel


class Policy(Base):
    def __init__(self,
                 envspec: EnvGroupArgs,
                 batch_size=128,
                 gamma=0.999,
                 max_train_step=1e18,
                 decay_lr=False,
                 normalize_vector_obs=False,
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
                 use_curiosity=False,
                 curiosity_reward_eta=0.01,
                 curiosity_lr=1.0e-3,
                 curiosity_beta=0.2,
                 **kwargs):
        super().__init__(**kwargs)

        self.envspec = envspec
        self.obs_spec = envspec.obs_spec
        self.is_continuous = envspec.is_continuous
        self.a_dim = envspec.a_dim
        self.n_copys = envspec.n_copys
        if self.n_copys <= 0:
            raise ValueError('agents num must larger than zero.')

        # self._normalize_vector_obs = normalize_vector_obs
        # self._running_average = SimpleRunningAverage(dim=self.obs_spec.total_vector_dim) if self._normalize_vector_obs else DefaultRunningAverage()

        self.batch_size = batch_size
        self.gamma = gamma
        self.train_step = 0
        self.max_train_step = max_train_step
        self.delay_lr = decay_lr

        self.representation_net_params = dict(representation_net_params)
        self.use_rnn = bool(self.representation_net_params.get('use_rnn', False))
        self.rep_net = DefaultRepresentationNetwork(obs_spec=self.obs_spec,
                                                    representation_net_params=self.representation_net_params).to(self.device)

        self.use_curiosity = use_curiosity
        if self.use_curiosity:
            self.curiosity_model = CuriosityModel(self.obs_spec,
                                                  self.representation_net_params,
                                                  self.is_continuous,
                                                  self.a_dim,
                                                  eta=curiosity_reward_eta,
                                                  lr=curiosity_lr,
                                                  beta=curiosity_beta)
            self._trainer_modules.update(curiosity_model=self.curiosity_model)

    # def normalize_vector_obs(self, x: np.ndarray) -> np.ndarray:
    #     return self._running_average.normalize(x)

    def reset(self) -> NoReturn:
        '''reset model for each new episode.'''
        self.cell_state = self.next_cell_state = self.initial_cell_state(batch=self.n_copys)

    def initial_cell_state(self, batch: int) -> Tuple[t.Tensor]:
        if self.use_rnn:
            return self.rep_net.memory_net.initial_cell_state(batch=batch)
        return None

    def get_cell_state(self) -> Tuple[Optional[t.Tensor]]:
        return self.cell_state

    def set_cell_state(self, cs: Tuple[Optional[t.Tensor]]) -> NoReturn:
        self.cell_state = cs

    def partial_reset(self, done: Union[List, np.ndarray]) -> NoReturn:
        self._partial_reset_cell_state(index=np.where(done)[0])

    def _partial_reset_cell_state(self, index: Union[List, np.ndarray]) -> NoReturn:
        '''
        根据环境的done的index，局部初始化RNN的隐藏状态
        '''
        assert isinstance(index, (list, np.ndarray)), 'assert isinstance(index, (list, np.ndarray))'
        if self.cell_state is not None and len(index) > 0:
            _arr = np.ones(shape=self.cell_state[0].shape, dtype=np.float32)    # h, c
            _arr[index] = 0.
            self.cell_state = [c * _arr for c in self.cell_state]        # [A, B] * [A, B] => [A, B] 将某行全部替换为0.

    def intermediate_variable_reset(self) -> NoReturn:
        '''
        TODO: Annotation
        '''
        self.summaries = {}

    @abstractmethod
    def __call__(self, obs, evaluation=False) -> Any:
        '''
        '''
        pass

    @abstractmethod
    def initialize_data_buffer(self, buffer) -> Any:
        '''
        TODO: Annotation
        '''
        pass
