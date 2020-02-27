import numpy as np
import tensorflow as tf
from Algorithms.tf2algos.base.policy import Policy
from utils.sth import sth
from typing import Dict


class Off_Policy(Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,
                 **kwargs):
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim_or_list=a_dim_or_list,
            is_continuous=is_continuous,
            **kwargs)
        self.batch_size = int(kwargs.get('batch_size', 128))
        self.buffer_size = int(kwargs.get('buffer_size', 10000))
        self.use_priority = kwargs.get('use_priority', False)
        self.n_step = kwargs.get('n_step', False)

    def set_buffer(self, buffer):
        '''
        TODO: Annotation
        '''
        self.data = buffer

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        """
        for off-policy training, use this function to store <s, a, r, s_, done> into ReplayBuffer.
        """
        assert isinstance(a, np.ndarray), "store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store need done type is np.ndarray"
        if not self.is_continuous:
            a = sth.action_index2one_hot(a, self.a_dim_or_list)
        self.data.add(
            s,
            visual_s,
            a,
            r[:, np.newaxis],
            s_,
            visual_s_,
            done[:, np.newaxis]
        )

    def no_op_store(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "no_op_store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "no_op_store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "no_op_store need done type is np.ndarray"
        if not self.is_continuous:
            a = sth.action_index2one_hot(a, self.a_dim_or_list)
        self.data.add(
            s,
            visual_s,
            a,
            r[:, np.newaxis],
            s_,
            visual_s_,
            done[:, np.newaxis]
        )

    def get_trainsitions(self, data_name_list=['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done']):
        '''
        TODO: Annotation
        '''
        return dict([
            [n, d] for n, d in zip(data_name_list, list(map(self.data_convert, self.data.sample())))
        ])

    def get_value_from_dict(self, data_name_list, data_dict):
        '''
        TODO: Annotation
        '''
        return [data_dict.get(n) for n in data_name_list]

    def _learn(self, function_dict: Dict):
        '''
        TODO: Annotation
        '''
        _pre_process = function_dict.get('pre_process_function', lambda *args: None)
        _train = function_dict.get('train_function', lambda *args: None)
        _update = function_dict.get('update_function', lambda *args: None)  # maybe need update parameters of target networks
        _summary = function_dict.get('summary_dict', {})
        _data_list = function_dict.get('data_list', ['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done'])

        if self.data.is_lg_batch_size:
            self.intermediate_variable_reset()
            data = self.get_trainsitions(data_name_list=_data_list)  # default: s, visual_s, a, r, s_, visual_s_, done
            _pre_process(data)

            if self.use_curiosity:
                ir, iloss, isummaries = self.curiosity_model(
                    *self.get_value_from_dict(data_name_list=['s', 'visual_s', 'a', 's_', 'visual_s_'], data_dict=data))
                data['r'] += ir
                self.curiosity_loss_constant += iloss
                self.summaries.update(isummaries)

            if self.use_priority:
                self.IS_w = self.data.get_IS_w()
            
            td_error, summaries = _train(*self.get_value_from_dict(data_name_list=_data_list, data_dict=data))
            self.summaries.update(summaries)

            if self.use_priority:
                td_error = np.squeeze(td_error.numpy())
                self.data.update(td_error, self.episode)
            _update()
            self.summaries.update(_summary)
            self.write_training_summaries(self.global_step, self.summaries)