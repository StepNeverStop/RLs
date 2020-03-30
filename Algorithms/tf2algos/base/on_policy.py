import numpy as np
import tensorflow as tf
from typing import Dict
from utils.sth import sth
from Algorithms.tf2algos.base.policy import Policy
from utils.on_policy_buffer import DataBuffer

class On_Policy(Policy):
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

    def initialize_data_buffer(self, data_name_list=['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done']):
        self.data = DataBuffer(dict_keys=data_name_list)

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        """
        for on-policy training, use this function to store <s, a, r, s_, done> into DataFrame of Pandas.
        """
        assert isinstance(a, np.ndarray), "store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store need done type is np.ndarray"
        self.data.add(s, visual_s, a, r, s_, visual_s_, done)

    def no_op_store(self, *args, **kwargs):
        pass

    def clear(self):
        """
        clear the DataFrame.
        """
        self.data.clear()

    def _learn(self, function_dict: Dict, epoch=1):
        '''
        TODO: Annotation
        '''
        _cal_stics = function_dict.get('calculate_statistics', lambda *args: None)
        _train = function_dict.get('train_function', lambda *args: None)    # 训练过程
        _train_data_list = function_dict.get('train_data_list', ['s', 'visual_s', 'a', 'discounted_reward', 'log_prob', 'gae_adv'])
        _summary = function_dict.get('summary_dict', {})    # 记录输出到tensorboard的词典

        self.intermediate_variable_reset()

        if self.use_curiosity:
            s, visual_s, a, r, s_, visual_s_ = self.data.get_curiosity_data()
            crsty_r, crsty_loss, crsty_summaries = self.curiosity_model(s, visual_s, a, s_, visual_s_)
            self.data.r = r.reshape([self.data.eps_len, -1])
            self.summaries.update(crsty_summaries)
        else:
            crsty_loss = tf.constant(value=0., dtype=self._data_type)

        _cal_stics()
        self.data.convert_action2one_hot(self.a_counts)

        for _ in range(epoch):
            all_data = self.data.sample_generater(self.batch_size, _train_data_list)
            for data in all_data:

                if self.use_rnn and self.burn_in_time_step:
                    raise NotImplementedError
                    # _s, _visual_s = self.data.get_burn_in_states()
                    # cell_state = self.get_burn_in_feature(_s, _visual_s)
                else:
                    cell_state = None

                data = list(map(self.data_convert, data))
                summaries = _train(data, crsty_loss, cell_state)

        self.summaries.update(summaries)
        self.summaries.update(_summary)

        self.write_training_summaries(self.episode, self.summaries)

        self.clear()
