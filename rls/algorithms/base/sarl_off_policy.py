#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from rls.algorithms.base.sarl_policy import SarlPolicy
from rls.common.data import Data
from rls.common.decorator import iton
from rls.common.when import Every
from rls.common.yaml_ops import load_config
from rls.utils.converter import to_numpy, to_tensor
from rls.utils.np_utils import int2one_hot


class SarlOffPolicy(SarlPolicy):

    def __init__(self,
                 epochs=1,
                 train_times=1,
                 chunk_length=1,
                 batch_size=256,
                 buffer_size=100000,
                 use_priority=False,
                 train_interval=1,
                 **kwargs):
        self._epochs = epochs
        self._train_times = train_times
        self._chunk_length = chunk_length
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.use_priority = use_priority
        self._should_train = Every(train_interval)
        super().__init__(**kwargs)

    def learn(self, BATCH: Data):
        BATCH = self._preprocess_BATCH(BATCH)
        td_errors = 0
        for _ in range(self._epochs):
            BATCH = self._before_train(BATCH)
            td_error, summaries = self._train(BATCH)
            td_errors += td_error  # [T, B, 1]
            self.summaries.update(summaries)
            self._after_train()
        return td_errors / self._epochs

    def episode_step(self,
                     obs: Data,
                     env_rets: Data,
                     begin_mask: np.ndarray):
        super().episode_step(obs, env_rets, begin_mask)
        if self._is_train_mode and self._buffer.can_sample and self._should_train(self._cur_interact_step):
            for _ in range(self._train_times):
                ret = self.learn(self._buffer.sample()[self._agent_id])
                if self.use_priority:
                    # td_error   [T, B, 1]
                    self._buffer.update(ret)

    # customed

    def _build_buffer(self):
        if self.use_priority:
            from rls.memories.per_buffer import PrioritizedDataBuffer
            buffer = PrioritizedDataBuffer(n_copies=self.n_copies,
                                           batch_size=self.batch_size,
                                           buffer_size=self.buffer_size,
                                           chunk_length=self._chunk_length,
                                           max_train_step=self._max_train_step,
                                           **load_config(f'rls/configs/buffer/off_policy_buffer.yaml')[
                                               'PrioritizedDataBuffer'])
        else:
            from rls.memories.er_buffer import DataBuffer
            buffer = DataBuffer(n_copies=self.n_copies,
                                batch_size=self.batch_size,
                                buffer_size=self.buffer_size,
                                chunk_length=self._chunk_length)
        return buffer

    def _preprocess_BATCH(self, BATCH):  # [T, B, *]
        if not self.is_continuous:
            # [T, B, 1] or [T, B] => [T, B, N]
            BATCH.action = int2one_hot(BATCH.action, self.a_dim)
        if self._obs_with_pre_action:
            BATCH.obs.update(other=np.concatenate((
                np.zeros_like(BATCH.action[:1]),  # TODO: improve
                BATCH.action[:-1]
            ), 0))
            BATCH.obs_.update(other=BATCH.action)
        return BATCH

    def _before_train(self, BATCH):
        self.summaries = {}
        if self.use_curiosity:
            crsty_r, crsty_summaries = self.curiosity_model(to_tensor(BATCH, device=self.device))
            BATCH.reward += to_numpy(crsty_r)
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
