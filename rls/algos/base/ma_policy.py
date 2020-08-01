#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from abc import abstractmethod
from typing import \
    List, \
    Dict, \
    Union, \
    Callable, \
    Any, \
    Optional, \
    NoReturn

from rls.algos.base.base import Base
from rls.nn.learningrate import ConsistentLearningRate
from rls.utils.list_utils import count_repeats


class MultiAgentPolicy(Base):
    def __init__(self,
                 s_dim: Union[List[int], np.ndarray],
                 visual_sources: Union[List[int], np.ndarray],
                 visual_resolution: Union[List, np.ndarray],
                 a_dim: Union[List[int], np.ndarray],
                 is_continuous: Union[List[bool], np.ndarray], ,
                 **kwargs):
        super().__init__(**kwargs)
        self.brain_controls = kwargs.get('brain_controls')
        self.s_dim = count_repeats(s_dim, self.brain_controls)
        self.visual_sources = count_repeats(visual_sources, self.brain_controls)    # not use yet

        self.batch_size = int(kwargs.get('batch_size', 128))
        self.n_agents = kwargs.get('n_agents', None)
        if not self.n_agents:
            raise ValueError('agents num is None.')

        self.is_continuous = count_repeats(is_continuous, self.brain_controls)
        self.a_dim = count_repeats(a_dim, self.brain_controls)
        self.gamma = float(kwargs.get('gamma', 0.999))
        self.train_step = 0
        self.max_train_step = int(kwargs.get('max_train_step', 1000))
        self.delay_lr = bool(kwargs.get('decay_lr', True))

        self.agent_sep_ctls = sum(self.brain_controls)
        self.writers = [self._create_writer(self.log_dir + f'_{i}') for i in range(self.agent_sep_ctls)]

    def init_lr(self, lr: float) -> Callable:
        if self.delay_lr:
            return tf.keras.optimizers.schedules.PolynomialDecay(lr, self.max_train_step, 1e-10, power=1.0)
        else:
            return ConsistentLearningRate(lr)

    def init_optimizer(self, lr: Callable, *args, **kwargs) -> tf.keras.optimizers.Optimizer:
        return tf.keras.optimizers.Adam(learning_rate=lr(self.train_step), *args, **kwargs)

    def reset(self) -> Any:
        pass

    def partial_reset(self, done: Union[List, np.ndarray]) -> Any:
        pass

    def intermediate_variable_reset(self) -> NoReturn:
        '''
        TODO: Annotation
        '''
        self.summaries = {}

    @abstractmethod
    def choose_actions(self, s, visual_s, evaluation: bool = False) -> Any:
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
    def _get_actions(self, s, visual_s, is_training: bool = True) -> Any:
        '''
        TODO: Annotation
        '''
        raise NotImplementedError

    def set_buffer(self, buffer) -> Any:
        '''
        TODO: Annotation
        '''
        pass

    def writer_summary(self, global_step: Union[int, tf.Variable], agent_idx: int = 0, **kargs) -> NoReturn:
        """
        record the data used to show in the tensorboard
        """
        super().writer_summary(global_step, writer=self.writers[agent_idx], **kargs)
