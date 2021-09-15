#!/usr/bin/env python3
# encoding: utf-8

from typing import Any, Dict, List, NoReturn, Union

import numpy as np
import torch as t

from rls.algorithms.base.sarl_policy import SarlPolicy
from rls.common.data import Data
from rls.common.decorator import iton
from rls.utils.np_utils import int2one_hot


class SarlOnPolicy(SarlPolicy):
    def __init__(self,
                 buffer_size,

                 epochs=4,
                 chunk_length=1,
                 batch_size=256,
                 sample_allow_repeat=True,
                 **kwargs):
        self._epochs = epochs
        self._chunk_length = chunk_length
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self._sample_allow_repeat = sample_allow_repeat
        super().__init__(**kwargs)

    def learn(self, BATCH: Data):
        BATCH = self._preprocess_BATCH(BATCH)   # [T, B, *]
        for _ in range(self._epochs):
            for _BATCH in BATCH.sample(self._chunk_length, self.batch_size, repeat=self._sample_allow_repeat):
                _BATCH = self._before_train(_BATCH)
                summaries = self._train(_BATCH)
                self.summaries.update(summaries)
                self._after_train()

    def episode_end(self):
        super().episode_end()
        if self._is_train_mode:
            # on-policy replay buffer
            self.learn(self._buffer.sample(0)[self._agent_id])
            self._buffer.clear()

    # customed

    def _build_buffer(self):
        from rls.memories import DataBuffer
        buffer = DataBuffer(n_copys=self.n_copys,
                            batch_size=self.batch_size,
                            buffer_size=self.buffer_size,
                            chunk_length=self._chunk_length)
        return buffer

    def _preprocess_BATCH(self, BATCH):  # [T, B, *]
        if not self.is_continuous:
            # [T, B, 1] or [T, B] => [T, B, N]
            BATCH.action = int2one_hot(
                BATCH.action, self.a_dim)
        if self._obs_with_pre_action:
            BATCH.obs.update(other=np.concatenate((
                np.zeros_like(BATCH.action[:1]),    # TODO: improve
                BATCH.action[:-1]
            ), 0))
            BATCH.obs_.update(other=BATCH.action)
        return BATCH

    def _before_train(self, BATCH):
        self.summaries = {}
        if self.use_curiosity:
            crsty_r, crsty_summaries = self.curiosity_model(
                to_tensor(BATCH, device=self.device))
            BATCH.reward += to_numpy(crsty_r)   # [T, B, 1]
            self.summaries.update(crsty_summaries)
        return BATCH

    @iton
    def _train(self, BATCH):
        raise NotImplementedError

    def _after_train(self):
        self._write_log(summaries=self.summaries,
                        step_type='step')
        if self._should_save_model(self._cur_train_step):
            self.save()
        self._cur_train_step += 1
