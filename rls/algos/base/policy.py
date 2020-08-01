#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from abc import abstractmethod
from typing import \
    Union, \
    List, \
    Callable, \
    Tuple, \
    Any, \
    Dict, \
    NoReturn, \
    Optional

from rls.algos.base.base import Base
from rls.nn.networks import CuriosityModel
from rls.nn.learningrate import ConsistentLearningRate
from rls.utils.vector_runing_average import \
    DefaultRunningAverage, \
    SimpleRunningAverage


class Policy(Base):
    def __init__(self,
                 s_dim: Union[int, np.ndarray],
                 visual_sources: Union[int, np.ndarray],
                 visual_resolution: Union[List, np.ndarray],
                 a_dim: Union[int, np.ndarray],
                 is_continuous: Union[bool, np.ndarray],
                 **kwargs):
        super().__init__(**kwargs)
        self.s_dim = s_dim
        self.feat_dim = self.s_dim
        self.visual_sources = visual_sources
        if visual_sources >= 1:
            self.use_visual = True
            self.visual_dim = [visual_sources, *visual_resolution]
        else:
            self.use_visual = False
            self.visual_dim = [0]

        self.use_rnn = bool(kwargs.get('use_rnn', False))

        self._normalize_vector_obs = bool(kwargs.get('normalize_vector_obs', False))
        self._running_average = SimpleRunningAverage(dim=self.s_dim) if self._normalize_vector_obs else DefaultRunningAverage()

        self.other_tv = []

        self.batch_size = int(kwargs.get('batch_size', 128))
        self.n_agents = int(kwargs.get('n_agents', 0))
        if self.n_agents <= 0:
            raise ValueError('agents num must larger than zero.')

        self.is_continuous = is_continuous
        self.a_dim = a_dim
        self.gamma = float(kwargs.get('gamma', 0.999))
        self.train_step = 0
        self.max_train_step = int(kwargs.get('max_train_step', 1000))
        self.delay_lr = bool(kwargs.get('decay_lr', True))

        self.use_curiosity = bool(kwargs.get('use_curiosity', False))
        if self.use_curiosity:
            self.curiosity_eta = float(kwargs.get('curiosity_reward_eta'))
            self.curiosity_lr = float(kwargs.get('curiosity_lr'))
            self.curiosity_beta = float(kwargs.get('curiosity_beta'))
            self.curiosity_loss_weight = float(kwargs.get('curiosity_loss_weight'))
            self.curiosity_model = CuriosityModel(self.is_continuous, self.s_dim, self.a_dim, self.visual_dim, 128,
                                                  eta=self.curiosity_eta, lr=self.curiosity_lr, beta=self.curiosity_beta, loss_weight=self.curiosity_loss_weight)
            self._worker_params_dict.update(curiosity_model=self.curiosity_model)
        self.writer = self._create_writer(self.log_dir)  # TODO: Annotation

    def init_lr(self, lr: float) -> Callable:
        if self.delay_lr:
            return tf.keras.optimizers.schedules.PolynomialDecay(lr, self.max_train_step, 1e-10, power=1.0)
        else:
            return ConsistentLearningRate(lr)

    def normalize_vector_obs(self, x: np.ndarray) -> np.ndarray:
        return self._running_average.normalize(x)

    def init_optimizer(self, lr: Callable, *args, **kwargs) -> tf.keras.optimizers.Optimizer:
        return tf.keras.optimizers.Adam(learning_rate=lr(self.train_step), *args, **kwargs)

    def reset(self) -> NoReturn:
        self.cell_state = (None,)

    def get_cell_state(self) -> Tuple:
        return self.cell_state

    def set_cell_state(self, cs) -> Any:
        pass

    def partial_reset(self, done: Union[List, np.ndarray]) -> Any:
        pass

    def intermediate_variable_reset(self) -> NoReturn:
        '''
        TODO: Annotation
        '''
        self.summaries = {}

    @abstractmethod
    def choose_action(self, s, visual_s, evaluation=False) -> Any:
        '''
        choose actions while training.
        Input: 
            s: vector observation
            visual_s: visual observation
        Output: 
            actions
        '''
        pass

    @tf.function
    def _get_action(self, s, visual_s, is_training: bool = True) -> Any:
        '''
        TODO: Annotation
        '''
        raise NotImplementedError

    def set_buffer(self, buffer) -> Any:
        '''
        TODO: Annotation
        '''
        pass
