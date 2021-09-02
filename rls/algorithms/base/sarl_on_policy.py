#!/usr/bin/env python3
# encoding: utf-8

from typing import Any, Dict, List, NoReturn, Union

import numpy as np
import torch as t

from rls.algorithms.base.sarl_policy import SarlPolicy
from rls.common.specs import Data
from rls.utils.np_utils import int2one_hot


class SarlOnPolicy(SarlPolicy):
    def __init__(self,
                 buffer_size,

                 epochs=4,
                 chunk_length=1,
                 batch_size=256,
                 **kwargs):
        self._epochs = epochs
        self._chunk_length = chunk_length
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        super().__init__(**kwargs)

    def learn(self, BATCH: Data):
        BATCH = self._preprocess_BATCH(BATCH)   # [T, B, *]
        for _ in range(self._epochs):
            for _BATCH in self._generate_BATCH(BATCH):
                _BATCH = self._before_train(_BATCH)
                summaries = self._train(_BATCH)
                self.summaries.update(summaries)
                self._after_train()

    def episode_end(self):
        super().episode_end()
        if self._is_train_mode \
                and self._buffer.can_sample:
            # on-policy replay buffer
            self.learn(self._buffer.all_data()[self._agent_id])
            self._buffer.clear()

    # customed

    def _build_buffer(self):
        from rls.memories.onpolicy_buffer import OnPolicyDataBuffer
        buffer = OnPolicyDataBuffer(n_copys=self.n_copys,
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

    def _generate_BATCH(self, BATCH, repeat=False):
        shape = BATCH.action.shape
        T = shape[0]
        B = shape[1]

        if repeat:
            for _ in range((T-self._chunk_length+1)*self.n_copys//self.batch_size):
                x = np.random.randint(
                    0, T - self._chunk_length + 1, self.batch_size)    # [B, ]
                y = np.random.randint(0, B, self.batch_size)  # (B, )
                xs = np.tile(
                    np.arange(self._chunk_length)[:, np.newaxis],
                    self.batch_size) + x  # (T, B) + (B, ) = (T, B)
                sample_idxs = (xs, y)
                yield BATCH[sample_idxs]
        else:
            # [N, ] + [B, 1] => [B, N]
            x = np.arange(0, T - self._chunk_length + 1, self._chunk_length) \
                + np.random.randint(0, T %
                                    self._chunk_length + 1, B)[:, np.newaxis]
            y = np.arange(B).repeat(x.shape[-1])   # [B*N]
            x = x.ravel()   # [B*N]
            idxs = np.arange(len(x))  # [B*N]
            np.random.shuffle(idxs)  # [B*N]
            for i in range(len(idxs)//self.batch_size):
                # [T, B]
                start, end = i*self.batch_size, (i+1)*self.batch_size
                xs = x[start:end] + \
                    np.tile(np.arange(self._chunk_length)[
                            :, np.newaxis], self.batch_size)
                sample_idxs = (xs, y[start:end])
                yield BATCH[sample_idxs]

    def _before_train(self, BATCH):
        self.summaries = {}
        if self.use_curiosity:
            crsty_r, crsty_summaries = self.curiosity_model(
                to_tensor(BATCH, device=self.device))
            BATCH.reward += to_numpy(crsty_r)   # [T, B, 1]
            self.summaries.update(crsty_summaries)
        return BATCH

    def _train(self, BATCH):
        raise NotImplementedError

    def _after_train(self):
        self._write_train_summaries(
            self.cur_train_step, self.summaries, self.writer)
        self.cur_train_step += 1
        if self.cur_train_step % self._save_frequency == 0:
            self.save()
