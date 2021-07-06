#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf

from typing import (Dict,
                    Union,
                    List,
                    NoReturn,
                    Any)

from rls.memories.on_policy_buffer import DataBuffer
from rls.algos.base.policy import Policy
from rls.utils.specs import BatchExperiences


class On_Policy(Policy):
    def __init__(self, envspec, **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.rnn_time_step = int(kwargs.get('rnn_time_step', 8))

    def initialize_data_buffer(self, store_data_type=BatchExperiences, sample_data_type=BatchExperiences) -> NoReturn:
        self.data = DataBuffer(n_copys=self.n_copys, rnn_cell_nums=self.rnn_cell_nums,
                               batch_size=self.batch_size, rnn_time_step=self.rnn_time_step,
                               store_data_type=store_data_type, sample_data_type=sample_data_type)

    def store_data(self, exps: BatchExperiences) -> NoReturn:
        """
        for on-policy training, use this function to store <s, a, r, s_, done> into DataBuffer.
        """
        # self._running_average()
        self.data.add(exps)
        if self.use_rnn:
            self.data.add_cell_state(tuple(cs.numpy() for cs in self.cell_state))
        self.cell_state = self.next_cell_state

    def no_op_store(self, *args, **kwargs) -> Any:
        pass

    def _learn(self, function_dict: Dict) -> NoReturn:
        '''
        TODO: Annotation
        '''
        _cal_stics = function_dict.get('calculate_statistics', lambda *args: None)
        _train = function_dict.get('train_function', lambda *args: None)    # 训练过程
        _summary = function_dict.get('summary_dict', {})    # 记录输出到tensorboard的词典

        self.intermediate_variable_reset()

        # self.data.normalize_vector_obs(self.normalize_vector_obs)

        if not self.is_continuous:
            self.data.convert_action2one_hot(self.a_dim)

        if self.use_curiosity and not self.use_rnn:
            curiosity_data = self.data.get_curiosity_data()
            curiosity_data.map_fn(self.data_convert)
            cell_states['obs'] = cell_states['obs_'] = self.initial_cell_state(batch=self.n_copys)
            crsty_r, crsty_summaries = self.curiosity_model(curiosity_data.nt, cell_states)
            self.data.update_reward(crsty_r.numpy())
            # self.data.r += crsty_r.numpy().reshape([self.data.eps_len, -1])
            self.summaries.update(crsty_summaries)

        _cal_stics()

        if self.use_rnn:
            all_data = self.data.sample_generater_rnn()
        else:
            all_data = self.data.sample_generater()

        for data, cell_state in all_data:
            data.map_fn(self.data_convert)
            cell_state = self.data_convert(cell_state)
            summaries = _train(data.nt, cell_state)

        self.summaries.update(summaries)
        self.summaries.update(_summary)

        self.write_training_summaries(self.train_step, self.summaries)

        self.data.clear()
