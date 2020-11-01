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


def make_on_policy_class(mode: str = 'share'):
    if mode == 'share':
        from rls.algos.base.share_rnn_cnn_policy import SharedPolicy as BasePolicy
    else:
        from rls.algos.base.policy import Policy as BasePolicy

    class On_Policy(BasePolicy):
        def __init__(self, envspec, **kwargs):
            super().__init__(envspec=envspec, **kwargs)
            self.rnn_time_step = int(kwargs.get('rnn_time_step', 8))

        def initialize_data_buffer(self,
                                   data_name_list: List[str] = ['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done']) -> NoReturn:
            self.data = DataBuffer(
                n_agents=self.n_agents,
                rnn_cell_nums=self.cell_state_nums,
                dict_keys=data_name_list
                )

        def store_data(self,
                       s: Union[List, np.ndarray],
                       visual_s: Union[List, np.ndarray],
                       a: Union[List, np.ndarray],
                       r: Union[List, np.ndarray],
                       s_: Union[List, np.ndarray],
                       visual_s_: Union[List, np.ndarray],
                       done: Union[List, np.ndarray]) -> NoReturn:
            """
            for on-policy training, use this function to store <s, a, r, s_, done> into DataBuffer.
            """
            assert isinstance(a, np.ndarray), "store need action type is np.ndarray"
            assert isinstance(r, np.ndarray), "store need reward type is np.ndarray"
            assert isinstance(done, np.ndarray), "store need done type is np.ndarray"
            self._running_average(s)
            data = (s, visual_s, a, r, s_, visual_s_, done)
            if self.use_rnn:
                data += tuple(cs.numpy() for cs in self.cell_state)
            self.data.add(*data)
            self.cell_state = self.next_cell_state

        def no_op_store(self, *args, **kwargs) -> Any:
            pass

        def clear(self) -> NoReturn:
            """
            clear the DataFrame.
            """
            self.data.clear()

        def _learn(self, function_dict: Dict) -> NoReturn:
            '''
            TODO: Annotation
            '''
            _cal_stics = function_dict.get('calculate_statistics', lambda *args: None)
            _train = function_dict.get('train_function', lambda *args: None)    # 训练过程
            _train_data_list = function_dict.get('train_data_list', ['s', 'visual_s', 'a', 'discounted_reward', 'log_prob', 'gae_adv'])
            _summary = function_dict.get('summary_dict', {})    # 记录输出到tensorboard的词典

            self.intermediate_variable_reset()

            self.data.normalize_vector_obs(self.normalize_vector_obs)

            if not self.is_continuous:
                self.data.convert_action2one_hot(self.a_dim)

            if self.use_curiosity:
                s, visual_s, a, r, s_, visual_s_ = self.data.get_curiosity_data()
                crsty_r, crsty_summaries = self.curiosity_model(s, visual_s, a, s_, visual_s_)
                self.data.r += crsty_r.numpy().reshape([self.data.eps_len, -1])
                self.summaries.update(crsty_summaries)

            _cal_stics()

            if self.use_rnn:
                all_data = self.data.sample_generater_rnn(self.batch_size, self.rnn_time_step, _train_data_list)
            else:
                all_data = self.data.sample_generater(self.batch_size, _train_data_list)

            for data in all_data:
                data = list(map(self.data_convert, data))
                summaries = _train(data)

            self.summaries.update(summaries)
            self.summaries.update(_summary)

            self.write_training_summaries(self.train_step, self.summaries)

            self.clear()

    return On_Policy
