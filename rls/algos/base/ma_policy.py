#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf

from abc import abstractmethod
from typing import (List,
                    Dict,
                    Union,
                    Callable,
                    Any,
                    Optional,
                    NoReturn)

from rls.algos.base.base import Base
from rls.nn.learningrate import ConsistentLearningRate
from rls.utils.list_utils import count_repeats
from rls.utils.indexs import MultiAgentEnvArgs


class MultiAgentPolicy(Base):
    def __init__(self, envspec: MultiAgentEnvArgs, **kwargs):
        super().__init__(**kwargs)
        self.group_controls = envspec.group_controls
        self.s_dim = count_repeats(envspec.s_dim, self.group_controls)
        self.visual_sources = count_repeats(envspec.visual_sources, self.group_controls)    # not use yet
        # self.visual_resolutions = envspec.visual_resolutions
        self.a_dim = count_repeats(envspec.a_dim, self.group_controls)
        self.is_continuous = count_repeats(envspec.is_continuous, self.group_controls)
        self.n_agents = envspec.n_agents
        if not self.n_agents:
            raise ValueError('agents num is None.')

        self.batch_size = int(kwargs.get('batch_size', 128))

        self.gamma = float(kwargs.get('gamma', 0.999))
        self.train_step = 0
        self.max_train_step = int(kwargs.get('max_train_step', 1000))
        self.delay_lr = bool(kwargs.get('decay_lr', True))

        self.agent_sep_ctls = sum(self.group_controls)
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
