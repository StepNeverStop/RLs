#!/usr/bin/env python3
# encoding: utf-8

import importlib
from abc import abstractmethod
from typing import Dict, List, NoReturn, Union

import numpy as np
import torch as t

from rls.algorithms.base.marl_policy import MarlPolicy
from rls.common.decorator import iTensor_oNumpy
from rls.common.specs import Data
from rls.common.yaml_ops import load_config
from rls.utils.np_utils import int2one_hot


class MultiAgentOffPolicy(MarlPolicy):

    def __init__(self,
                 n_time_step=4,
                 epochs=1,
                 batch_size=256,
                 buffer_size=100000,
                 use_priority=False,
                 **kwargs):
        self.n_time_step = n_time_step
        self.epochs = epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.use_priority = use_priority
        super().__init__(**kwargs)

    def _build_buffer(self):
        if self.use_priority == True:
            from rls.memories.per_buffer import PrioritizedDataBuffer
            buffer = PrioritizedDataBuffer(n_copys=self.n_copys,
                                           batch_size=self.batch_size,
                                           buffer_size=self.buffer_size,
                                           time_step=self.n_time_step,
                                           max_train_step=self.max_train_step,
                                           **load_config(f'rls/configs/buffer/off_policy_buffer.yaml')['PrioritizedDataBuffer'])
        else:
            from rls.memories.er_buffer import DataBuffer
            buffer = DataBuffer(n_copys=self.n_copys,
                                batch_size=self.batch_size,
                                buffer_size=self.buffer_size,
                                time_step=self.n_time_step)
        return buffer

    def episode_step(self, obs, env_rets: Dict[str, Data]):
        super().episode_step(obs, env_rets)
        if self._is_train_mode and self._buffer.can_sample:
            ret = self.learn(self._buffer.sample())
            if self.use_priority:
                # td_error   [T, B, 1]
                self._buffer.update(ret)

    def learn(self, BATCH_DICT: Data):
        BATCH_DICT = self._preprocess_BATCH(BATCH_DICT)
        td_errors = 0.
        for _ in range(self.epochs):
            BATCH_DICT = self._before_train(BATCH_DICT)
            td_error, summaries = self._train(BATCH_DICT)
            td_errors += td_error  # [T, B, 1]
            self.summaries.update(summaries)
            self._after_train()
        return td_errors/self.epochs

    # customed

    def _preprocess_BATCH(self, BATCH_DICT):  # [B, *] or [T, B, *]
        for id in self.agent_ids:
            if not self.is_continuouss[id]:
                shape = BATCH_DICT[id].action.shape
                # [T, B, 1] or [T, B] => [T, B, N]
                BATCH_DICT[id].action = int2one_hot(
                    BATCH_DICT[id].action, self.a_dims[id])
        for i, id in enumerate(self.agent_ids):
            other, other_ = None, None
            if self._obs_with_pre_action:
                other = np.concatenate((
                    np.zeros_like(BATCH_DICT[id].action[:1]),
                    BATCH_DICT[id].action[:-1]
                ), 0)
                other_ = BATCH_DICT[id].action
            if self._obs_with_agent_id:
                _id_onehot = int2one_hot(
                    np.full(BATCH_DICT[id].action.shape[:-1], i), self.n_agents_percopy)
                if other is not None:
                    other = np.concatenate((
                        other,
                        _id_onehot
                    ), -1)
                    other_ = np.concatenate((
                        other_,
                        _id_onehot
                    ), -1)
                else:
                    other, other_ = _id_onehot, _id_onehot
            BATCH_DICT[id].obs.update(other=other)
            BATCH_DICT[id].obs_.update(other=other_)
        return BATCH_DICT

    def _before_train(self, BATCH_DICT):
        self.summaries = {}
        return BATCH_DICT

    @iTensor_oNumpy
    def _train(self, BATCH_DICT):
        raise NotImplementedError

    def _after_train(self):
        self._write_train_summaries(self.cur_train_step, self.summaries)
        self.cur_train_step += 1
        if self.cur_train_step % self._save_frequency == 0:
            self.save()
