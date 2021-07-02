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
from rls.utils.specs import (EnvGroupArgs,
                             VectorNetworkType,
                             VisualNetworkType,
                             MemoryNetworkType)


class MultiAgentPolicy(Base):
    def __init__(self, envspecs: List[EnvGroupArgs], **kwargs):
        super().__init__(**kwargs)

        self.envspecs = envspecs
        self.n_copys = envspecs[0].n_copys
        self.n_agents_percopy = len(envspecs)

        self.batch_size = int(kwargs.get('batch_size', 128))
        self.gamma = float(kwargs.get('gamma', 0.999))
        self.train_step = 0
        self.max_train_step = int(kwargs.get('max_train_step', 1000))
        self.delay_lr = bool(kwargs.get('decay_lr', True))

        self.vector_net_kwargs = dict(kwargs.get('vector_net_kwargs', {}))
        self.vector_net_kwargs['network_type'] = VectorNetworkType(self.vector_net_kwargs['network_type'])

        self.visual_net_kwargs = dict(kwargs.get('visual_net_kwargs', {}))
        self.visual_net_kwargs['network_type'] = VisualNetworkType(self.visual_net_kwargs['network_type'])

        self.encoder_net_kwargs = dict(kwargs.get('encoder_net_kwargs', {}))

        self.memory_net_kwargs = dict(kwargs.get('memory_net_kwargs', {}))
        self.memory_net_kwargs['network_type'] = MemoryNetworkType(self.memory_net_kwargs['network_type'])

        self.writers = [self._create_writer(self.log_dir + f'_{i}') for i in range(self.n_agents_percopy)]

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

    @tf.function
    def _get_actions(self, obs, is_training: bool = True) -> Any:
        '''
        TODO: Annotation
        '''
        raise NotImplementedError

    def writer_summary(self, global_step: Union[int, tf.Variable], summaries) -> NoReturn:
        """
        record the data used to show in the tensorboard
        """
        for i, summary in enumerate(summaries):
            super().writer_summary(global_step, summaries=summary, writer=self.writers[i])

    def write_training_summaries(self,
                                 global_step: Union[int, tf.Variable],
                                 summaries: Dict,
                                 writer: Optional[tf.summary.SummaryWriter] = None) -> NoReturn:
        '''
        write tf summaries showing in tensorboard.
        '''
        super().write_training_summaries(global_step, summaries=summaries.get('model', {}), writer=self.writer)
        for i, summary in summaries.items():
            super().write_training_summaries(global_step, summaries=summary, writer=self.writers[i])
