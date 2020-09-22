#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from typing import (List,
                    Union,
                    NoReturn)

from rls.algos.base.ma_policy import MultiAgentPolicy


class MultiAgentOffPolicy(MultiAgentPolicy):
    def __init__(self, envspec, **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.total_s_dim = sum(self.s_dim)
        self.total_a_dim = sum(self.a_dim)

        self.buffer_size = int(kwargs.get('buffer_size', 10000))
        self.n_step = kwargs.get('n_step', False)
        self.train_times_per_step = int(kwargs.get('train_times_per_step', 1))

    def set_buffer(self, buffer) -> NoReturn:
        '''
        TODO: Annotation
        '''
        self.data = buffer

    def store_data(self, *args) -> NoReturn:
        """
        args: [
            batch vector_obs of agent NO.1, ..., batch vector_obs of agent NO.n, 
            batch visual_obs of agent NO.1, ..., batch visual_obs of agent NO.n,
            batch action of agent NO.1, ..., batch action of agent NO.n,
            batch reward of agent NO.1, ..., batch reward of agent NO.n,
            batch next vector_obs of agent NO.1, ..., batch next vector_obs of agent NO.n,
            batch next visual_obs of agent NO.1, ..., batch next visual_obs of agent NO.n,
            batch done
            ]
        """
        self.data.add(*args)

    def no_op_store(self, *args) -> NoReturn:
        '''
        args: [
            batch vector_obs of agent NO.1, ..., batch vector_obs of agent NO.n, 
            batch visual_obs of agent NO.1, ..., batch visual_obs of agent NO.n,
            batch action of agent NO.1, ..., batch action of agent NO.n,
            batch reward of agent NO.1, ..., batch reward of agent NO.n,
            batch next vector_obs of agent NO.1, ..., batch next vector_obs of agent NO.n,
            batch next visual_obs of agent NO.1, ..., batch next visual_obs of agent NO.n,
            batch done
            ]
        '''
        self.data.add(*args)
