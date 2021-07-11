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

from rls.algos.base.base import Base
from rls.common.specs import EnvGroupArgs


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

        self.representation_net_params = dict(kwargs.get('representation_net_params', defaultdict(dict)))

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

    def writer_summary(self, global_step: Union[int, t.Tensor], summaries) -> NoReturn:
        """
        record the data used to show in the tensorboard
        """
        for i, summary in enumerate(summaries):
            super().writer_summary(global_step, summaries=summary, writer=self.writers[i])

    def write_training_summaries(self,
                                 global_step: Union[int, t.Tensor],
                                 summaries: Dict,
                                 writer=None) -> NoReturn:
        '''
        write tf summaries showing in tensorboard.
        '''
        super().write_training_summaries(global_step, summaries=summaries.get('model', {}), writer=self.writer)
        for i, summary in summaries.items():
            if i != 'model':  # TODO: Optimization
                super().write_training_summaries(global_step, summaries=summary, writer=self.writers[i])
