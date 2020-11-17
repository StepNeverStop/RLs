#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf

from abc import abstractmethod
from typing import (Union,
                    List,
                    Callable,
                    Tuple,
                    Any,
                    Dict,
                    NoReturn,
                    Optional)

from rls.algos.base.base import Base
from rls.nn.learningrate import ConsistentLearningRate
from rls.utils.vector_runing_average import (DefaultRunningAverage,
                                             SimpleRunningAverage)
from rls.utils.indexs import (SingleAgentEnvArgs,
                              VisualNetworkType,
                              MemoryNetworkType)
from rls.utils.build_networks import DefaultRepresentationNetwork
from rls.nn.modules import CuriosityModel


class Policy(Base):
    def __init__(self, envspec: SingleAgentEnvArgs, **kwargs):
        super().__init__(**kwargs)

        # TODO: 分离状态输入
        self.s_dim = envspec.s_dim
        self.visual_sources = envspec.visual_sources
        self.visual_resolutions = envspec.visual_resolutions
        self.is_continuous = envspec.is_continuous
        self.a_dim = envspec.a_dim
        self.n_agents = envspec.n_agents
        if self.n_agents <= 0:
            raise ValueError('agents num must larger than zero.')

        # TODO: 分离向量输入
        self.vector_dims = [self.s_dim] if self.s_dim > 0 else []
        self.visual_dims = self.visual_sources * [self.visual_resolutions]

        self._normalize_vector_obs = bool(kwargs.get('normalize_vector_obs', False))
        self._running_average = SimpleRunningAverage(dim=self.s_dim) if self._normalize_vector_obs else DefaultRunningAverage()

        self.batch_size = int(kwargs.get('batch_size', 128))
        self.gamma = float(kwargs.get('gamma', 0.999))
        self.train_step = 0
        self.max_train_step = int(kwargs.get('max_train_step', 1000))
        self.delay_lr = bool(kwargs.get('decay_lr', True))

        self.vector_net_kwargs = dict(kwargs.get('vector_net_kwargs', {}))

        self.visual_net_kwargs = dict(kwargs.get('visual_net_kwargs', {}))
        self.visual_net_kwargs['network_type'] = VisualNetworkType(self.visual_net_kwargs['network_type'])

        self.encoder_net_kwargs = dict(kwargs.get('encoder_net_kwargs', {}))

        self.memory_net_kwargs = dict(kwargs.get('memory_net_kwargs', {}))
        self.rnn_type = self.memory_net_kwargs['network_type'] = MemoryNetworkType(self.memory_net_kwargs['network_type'])
        self.use_rnn = bool(self.memory_net_kwargs.get('use_rnn', False))
        self.cell_nums = 2 if self.rnn_type == MemoryNetworkType.LSTM else 1

        self._representation_net = self._create_representation_net(name='_representation_net')

        self.use_curiosity = bool(kwargs.get('use_curiosity', False))
        if self.use_curiosity:
            self.curiosity_eta = float(kwargs.get('curiosity_reward_eta'))
            self.curiosity_lr = float(kwargs.get('curiosity_lr'))
            self.curiosity_beta = float(kwargs.get('curiosity_beta'))
            self.curiosity_loss_weight = float(kwargs.get('curiosity_loss_weight'))
            self.curiosity_model = CuriosityModel(self.vector_dims,
                                                  self.visual_dims,
                                                  self.vector_net_kwargs,
                                                  self.visual_net_kwargs,
                                                  self.encoder_net_kwargs,
                                                  self.memory_net_kwargs,
                                                  self.is_continuous,
                                                  self.a_dim,
                                                  eta=self.curiosity_eta,
                                                  lr=self.curiosity_lr,
                                                  beta=self.curiosity_beta,
                                                  loss_weight=self.curiosity_loss_weight)
            self._worker_params_dict.update(curiosity_model=self.curiosity_model)

    def _create_representation_net(self, name: str = 'default'):
        # TODO: Added changeable command
        return DefaultRepresentationNetwork(name=name,
                                            vec_dims=self.vector_dims,
                                            vis_dims=self.visual_dims,
                                            vector_net_kwargs=self.vector_net_kwargs,
                                            visual_net_kwargs=self.visual_net_kwargs,
                                            encoder_net_kwargs=self.encoder_net_kwargs,
                                            memory_net_kwargs=self.memory_net_kwargs)

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
        '''reset model for each new episode.'''
        self.cell_state = self.next_cell_state = self.initial_cell_state(batch=self.n_agents)

    def initial_cell_state(self, batch: int) -> Tuple[tf.Tensor]:
        if self.use_rnn:
            return tuple(tf.zeros((batch, self._representation_net.h_dim), dtype=tf.float32) for _ in range(self.cell_nums))
        return (None,)

    def get_cell_state(self) -> Tuple[Optional[tf.Tensor]]:
        return self.cell_state

    def set_cell_state(self, cs: Tuple[Optional[tf.Tensor]]) -> NoReturn:
        self.cell_state = cs

    def partial_reset(self, done: Union[List, np.ndarray]) -> NoReturn:
        self._partial_reset_cell_state(index=np.where(done)[0])

    def _partial_reset_cell_state(self, index: Union[List, np.ndarray]) -> NoReturn:
        '''
        根据环境的done的index，局部初始化RNN的隐藏状态
        '''
        assert isinstance(index, (list, np.ndarray))
        if self.cell_state[0] is not None and len(index) > 0:
            _arr = np.ones(shape=self.cell_state[0].shape, dtype=np.float32)    # h, c
            _arr[index] = 0.
            self.cell_state = [c * _arr for c in self.cell_state]        # [A, B] * [A, B] => [A, B] 将某行全部替换为0.

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
