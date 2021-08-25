#!/usr/bin/env python3
# encoding: utf-8

import importlib
import numpy as np
import torch as t

from typing import (Dict,
                    Union,
                    NoReturn,
                    List,
                    Tuple)

from rls.utils.np_utils import int2one_hot
from rls.algorithms.base.sarl_policy import SarlPolicy
from rls.common.yaml_ops import load_config
from rls.common.specs import Data
from rls.common.decorator import iTensor_oNumpy
from rls.utils.converter import (to_numpy,
                                 to_tensor)


class SarlOffPolicy(SarlPolicy):
    def __init__(self,
                 epochs=1,
                 n_time_step=1,
                 batch_size=256,
                 buffer_size=100000,
                 use_priority=False,
                 **kwargs):
        self.epochs = epochs
        self.n_time_step = n_time_step
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.use_priority = use_priority
        super().__init__(**kwargs)

    def learn(self, BATCH: Data):
        BATCH = self._preprocess_BATCH(BATCH)
        td_errors = 0
        for _ in range(self.epochs):
            BATCH = self._before_train(BATCH)
            td_error, summaries = self._train(BATCH)
            td_errors += td_error   # [T, B, 1]
            self.summaries.update(summaries)
            self._after_train()
        return td_errors / self.epochs

    # customed

    def _preprocess_BATCH(self, BATCH):  # [T, B, *]
        if not self.is_continuous:
            # [T, B, 1] or [T, B] => [T, B, N]
            BATCH.action = int2one_hot(BATCH.action, self.a_dim)
        return BATCH

    def _before_train(self, BATCH):
        self.summaries = {}
        if self.use_curiosity:
            crsty_r, crsty_summaries = self.curiosity_model(
                to_tensor(BATCH, device=self.device))
            BATCH.reward += to_numpy(crsty_r)
            self.summaries.update(crsty_summaries)
        return BATCH

    @iTensor_oNumpy
    def _train(self, BATCH):
        raise NotImplementedError

    def _after_train(self):
        self._write_train_summaries(self.cur_train_step, self.summaries, self.writer)
        self.cur_train_step += 1
        if self.cur_train_step % self.save_frequency == 0:
            self.save()
