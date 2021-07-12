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
from rls.algos.base.policy import Policy
from rls.common.yaml_ops import load_yaml
from rls.common.specs import BatchExperiences


class Off_Policy(Policy):
    def __init__(self, envspec, **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.buffer_size = int(kwargs.get('buffer_size', 10000))

        self.n_step = int(kwargs.get('n_step', 1))

        self.use_priority = kwargs.get('use_priority', False)
        self.use_isw = bool(kwargs.get('use_isw', False))

        self.burn_in_time_step = int(kwargs.get('burn_in_time_step', 10))
        self.train_time_step = int(kwargs.get('train_time_step', 10))
        self.episode_batch_size = int(kwargs.get('episode_batch_size', 32))
        self.episode_buffer_size = int(kwargs.get('episode_buffer_size', 10000))

        self.train_times_per_step = int(kwargs.get('train_times_per_step', 1))

    def initialize_data_buffer(self) -> NoReturn:
        '''
        TODO: Annotation
        '''
        _buffer_args = {}
        if self.use_rnn:
            _type = 'EpisodeExperienceReplay'
            _buffer_args.update(
                batch_size=self.episode_batch_size,
                capacity=self.episode_buffer_size,
                burn_in_time_step=self.burn_in_time_step,
                train_time_step=self.train_time_step,
                n_copys=self.n_copys
            )
        else:
            _type = 'ExperienceReplay'
            _buffer_args.update(
                batch_size=self.batch_size,
                capacity=self.buffer_size
            )
            if self.use_priority:
                _type = 'Prioritized' + _type
                _buffer_args.update(
                    max_train_step=self.max_train_step
                )
            if self.n_step > 1:
                _type = 'NStep' + _type
                _buffer_args.update(
                    n_step=self.n_step,
                    gamma=self.gamma,
                    n_copys=self.n_copys
                )
                self.gamma = self.gamma ** self.n_step

        default_buffer_args = load_yaml(f'rls/configs/off_policy_buffer.yaml')[_type]
        default_buffer_args.update(_buffer_args)

        Buffer = getattr(importlib.import_module(f'rls.memories.single_replay_buffers'), _type)
        self.data = Buffer(**default_buffer_args)

    def store_data(self, exps: BatchExperiences) -> NoReturn:
        """
        for off-policy training, use this function to store <s, a, r, s_, done> into ReplayBuffer.
        """
        # self._running_average()
        self.data.add(exps)

    def no_op_store(self, exps: BatchExperiences) -> NoReturn:
        # self._running_average()
        self.data.add(exps)

    def get_transitions(self) -> BatchExperiences:
        '''
        TODO: Annotation
        '''
        exps = self.data.sample()   # 经验池取数据
        return self._data_process2dict(exps)

    def get_burn_in_transitions(self) -> BatchExperiences:
        exps = self.data.get_burn_in_data()
        return self._data_process2dict(exps)

    def _data_process2dict(self, exps: BatchExperiences) -> BatchExperiences:
        # TODO 优化
        if not self.is_continuous:
            exps.action = int2one_hot(exps.action.astype(np.int32), self.a_dim)
        # exps.obs.vector=self.normalize_vector_obs()
        # exps.obs_.vector=self.normalize_vector_obs()
        return exps

    def _train(self, *args):
        '''
        NOTE: usually need to override this function
        TODO: Annotation
        '''
        return (None, {})

    def _target_params_update(self, *args):
        '''
        NOTE: usually need to override this function
        TODO: Annotation
        '''
        return None

    def _learn(self, function_dict: Dict = {}) -> NoReturn:
        '''
        TODO: Annotation
        '''
        _summary = function_dict.get('summary_dict', {})    # 记录输出到tensorboard的词典

        if self.data.can_sample:
            self.intermediate_variable_reset()
            data = self.get_transitions()
            cell_states = {}

            # --------------------------------------burn in隐状态部分
            cell_states['obs'] = self.initial_cell_state(batch=self.episode_batch_size)
            cell_states['obs_'] = self.initial_cell_state(batch=self.episode_batch_size)
            if self.use_rnn and self.burn_in_time_step > 0:
                _burn_in_data = self.get_burn_in_transitions()
                _, cell_states['obs'] = self.rep_net(obs=_burn_in_data.obs,
                                                     cell_state=cell_states['obs'])
                _, cell_states['obs_'] = self.rep_net(obs=_burn_in_data.obs_,
                                                      cell_state=cell_states['obs_'])
            # --------------------------------------

            # --------------------------------------好奇心部分
            if self.use_curiosity:
                # TODO: check
                crsty_r, crsty_summaries = self.curiosity_model(data, cell_states)
                data.reward += crsty_r
                _summary.update(crsty_summaries)
            # --------------------------------------

            # --------------------------------------优先经验回放部分，获取重要性比例
            if self.use_priority and self.use_isw:
                _isw = self.data.get_IS_w().reshape(-1, 1)  # [B, ] => [B, 1]
            else:
                _isw = 1.
            # --------------------------------------

            # --------------------------------------训练主程序，返回可能用于PER权重更新的TD error，和需要输出tensorboard的信息
            td_error, summaries = self._train(data, _isw, cell_states)
            # --------------------------------------

            # --------------------------------------更新summary
            _summary.update(summaries)
            # --------------------------------------

            # --------------------------------------优先经验回放的更新部分
            if self.use_priority:
                td_error = np.squeeze(td_error.numpy())
                self.data.update(td_error)
            # --------------------------------------

            # --------------------------------------target网络的更新部分
            self._target_params_update()
            # --------------------------------------

            # --------------------------------------更新summary
            self.summaries.update(_summary)
            # --------------------------------------

            # --------------------------------------写summary到tensorboard
            self.write_training_summaries(self.global_step, self.summaries)
            # --------------------------------------

    def _apex_learn(self, function_dict: Dict, data: BatchExperiences, priorities) -> np.ndarray:
        '''
        TODO: Annotation
        '''
        _summary = function_dict.get('summary_dict', {})    # 记录输出到tensorboard的词典

        self.intermediate_variable_reset()
        data = self._data_process2dict(data=data)

        cell_state = None

        if self.use_curiosity:
            crsty_r, crsty_summaries = self.curiosity_model(data, cell_state)
            data.reward += crsty_r
            _summary.update(crsty_summaries)

        td_error, summaries = self._train(data, _isw, cell_state)
        _summary.update(summaries)

        self._target_params_update()
        self.summaries.update(_summary)
        self.write_training_summaries(self.global_step, self.summaries)

        return np.squeeze(td_error.numpy())

    def _apex_cal_td(self, data: BatchExperiences, function_dict: Dict = {}) -> np.ndarray:
        '''
        TODO: Annotation
        '''
        data = self._data_process2dict(data=data)

        cell_state = None
        td_error = self._cal_td(data, cell_state)
        return np.squeeze(td_error.numpy())
