#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf

from typing import (Dict,
                    Union,
                    NoReturn,
                    List,
                    Tuple)

from rls.utils.np_utils import int2one_hot
from rls.algos.base.policy import Policy
from rls.utils.specs import (MemoryNetworkType,
                             BatchExperiences,
                             NamedTupleStaticClass)


class Off_Policy(Policy):
    def __init__(self, envspec, **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.buffer_size = int(kwargs.get('buffer_size', 10000))
        self.use_priority = kwargs.get('use_priority', False)
        self.n_step = kwargs.get('n_step', False)
        self.use_isw = bool(kwargs.get('use_isw', False))
        self.train_times_per_step = int(kwargs.get('train_times_per_step', 1))

        self.burn_in_time_step = int(kwargs.get('burn_in_time_step', 20))
        self.episode_batch_size = int(kwargs.get('episode_batch_size', 32))

    def set_buffer(self, buffer) -> NoReturn:
        '''
        TODO: Annotation
        '''
        self.data = buffer

    def store_data(self, exps: BatchExperiences) -> NoReturn:
        """
        for off-policy training, use this function to store <s, a, r, s_, done> into ReplayBuffer.
        """
        self._running_average(exps.obs.vector)
        exps = exps._replace(reward=exps.reward[:, np.newaxis], done=exps.done[:, np.newaxis])
        self.data.add(exps)

    def no_op_store(self, exps: BatchExperiences) -> NoReturn:
        self._running_average(exps.obs.vector)
        exps = exps._replace(reward=exps.reward[:, np.newaxis], done=exps.done[:, np.newaxis])
        self.data.add(exps)

    def get_transitions(self) -> BatchExperiences:
        '''
        TODO: Annotation
        '''
        exps = self.data.sample()   # 经验池取数据
        return self._data_process2dict(exps)

    def _data_process2dict(self, exps: BatchExperiences) -> BatchExperiences:
        # TODO 优化
        if not self.is_continuous:
            assert 'action' in exps._fields, "assert 'action' in exps._fields"
            exps = exps._replace(action=int2one_hot(exps.action.astype(np.int32), self.a_dim))
        assert 'obs' in exps._fields and 'obs_' in exps._fields, "'obs' in exps._fields and 'obs_' in exps._fields"
        exps = exps._replace(
            obs=exps.obs._replace(vector=self.normalize_vector_obs(exps.obs.vector)),
            obs_=exps.obs_._replace(vector=self.normalize_vector_obs(exps.obs_.vector)))
        return NamedTupleStaticClass.data_convert(self.data_convert, exps)

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

    def _learn(self, function_dict: Dict) -> NoReturn:
        '''
        TODO: Annotation
        '''
        _summary = function_dict.get('summary_dict', {})    # 记录输出到tensorboard的词典
        _use_stack = function_dict.get('use_stack', False)

        if self.data.is_lg_batch_size:
            self.intermediate_variable_reset()
            data = self.get_transitions()

            # --------------------------------------burn in隐状态部分
            cell_state = self.initial_cell_state(batch=self.episode_batch_size)
            if self.use_rnn and self.burn_in_time_step > 0:
                _burn_in_data = self.data.get_burn_in_data()
                _, cell_state = self._representation_net(_burn_in_data.obs, cell_state)
            # --------------------------------------

            # --------------------------------------好奇心部分
            if self.use_curiosity:
                curiosity_data = data
                if self.use_rnn:
                    # TODO check visual
                    obs = [tf.reshape(o, [-1, o.shape[-1]]) for o in data.obs] # [B, T, N] => [B*T, N]
                    obs_ = [tf.reshape(o, [-1, o.shape[-1]]) for o in data.obs_]
                    curiosity_data = data._replace(obs=data.obs.__class__._make(obs), 
                                                   obs_=data.obs_.__class__._make(obs_))
                crsty_r, crsty_summaries = self.curiosity_model(curiosity_data, cell_state)
                data = data._replace(reward=data.reward+crsty_r)
                _summary.update(crsty_summaries)
            # --------------------------------------

            # --------------------------------------优先经验回放部分，获取重要性比例
            if self.use_priority and self.use_isw:
                _isw = self.data.get_IS_w().reshape(-1, 1)  # [B, ] => [B, 1]
                _isw = self.data_convert(_isw)
            else:
                _isw = tf.constant(value=1., dtype=self._tf_data_type)
            # --------------------------------------

            # --------------------------------------如果使用RNN， 就将s和s‘状态进行拼接处理
            if _use_stack:
                if self.use_rnn:
                    obs = [tf.concat([o, o_[:, -1:]], axis=1) for o, o_ in zip(data.obs, data.obs_)] # [B, T, N], [B, T, N] => [B, T+1, N]
                else:
                    obs = [tf.concat([o, o_], axis=0) for o, o_ in zip(data.obs, data.obs_)] # [B, N] => [2*B, N]
                data = data._replace(obs=data.obs.__class__._make(obs))
            # --------------------------------------

            # --------------------------------------训练主程序，返回可能用于PER权重更新的TD error，和需要输出tensorboard的信息
            td_error, summaries = self._train(data, _isw, cell_state)
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
        _use_stack = function_dict.get('use_stack', False)

        self.intermediate_variable_reset()
        data = self._data_process2dict(data=data)

        if _use_stack:
            obs = [tf.concat([o, o_], axis=0) for o, o_ in zip(data.obs, data.obs_)] # [B, N] => [2*B, N]
            data = data._replace(obs=data.obs.__class__._make(obs))

        cell_state = (None,)
        
        if self.use_curiosity:
            crsty_r, crsty_summaries = self.curiosity_model(data, cell_state)
            data = data._replace(reward=data.reward+crsty_r)
            _summary.update(crsty_summaries)

        _isw = self.data_convert(priorities)

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
        _use_stack = function_dict.get('use_stack', False)

        data = self._data_process2dict(data=data)

        if _use_stack:
            obs = [tf.concat([o, o_], axis=0) for o, o_ in zip(data.obs, data.obs_)] # [B, N] => [2*B, N]
            data = data._replace(obs=data.obs.__class__._make(obs))

        cell_state = (None,)
        td_error = self._cal_td(data, cell_state)
        return np.squeeze(td_error.numpy())
