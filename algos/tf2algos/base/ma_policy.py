#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from .base import Base
from abc import abstractmethod
from rls.learningrate import ConsistentLearningRate
from utils.list_utils import count_repeats


class MultiAgentPolicy(Base):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim,
                 is_continuous,
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

    def init_lr(self, lr):
        if self.delay_lr:
            return tf.keras.optimizers.schedules.PolynomialDecay(lr, self.max_train_step, 1e-10, power=1.0)
        else:
            return ConsistentLearningRate(lr)

    def init_optimizer(self, lr, *args, **kwargs):
        return tf.keras.optimizers.Adam(learning_rate=lr(self.train_step), *args, **kwargs)

    def reset(self):
        pass

    def partial_reset(self, done):
        pass

    def model_recorder(self, kwargs):
        kwargs.update(dict(global_step=self.global_step))
        self._create_saver(kwargs)
        self.show_logo()

    def intermediate_variable_reset(self):
        '''
        TODO: Annotation
        '''
        self.summaries = {}

    @abstractmethod
    def choose_actions(self, s, visual_s, evaluation=False):
        '''
        choose actions while training.
        Input: 
            s: vector observation
            visual_s: visual observation
        Output: 
            actions
        '''
        pass

    def update_target_net_weights(self, tge, src, ployak=None):
        '''
        update weights of target neural network.
        '''
        if ployak is None:
            tf.group([t.assign(s) for t, s in zip(tge, src)])
        else:
            tf.group([t.assign(ployak * t + (1 - ployak) * s) for t, s in zip(tge, src)])

    @tf.function
    def _get_actions(self, s, visual_s, is_training=True):
        '''
        TODO: Annotation
        '''
        raise NotImplementedError

    def set_buffer(self, buffer):
        '''
        TODO: Annotation
        '''
        pass

    def writer_summary(self, global_step, agent_idx=0, **kargs):
        """
        record the data used to show in the tensorboard
        """
        super().writer_summary(global_step, writer=self.writers[agent_idx], **kargs)
