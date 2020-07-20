import numpy as np
import tensorflow as tf
from utils.sth import sth
from typing import Dict


def make_off_policy_class(mode='share'):
    if mode == 'share':
        from algos.tf2algos.base.share_rnn_cnn_policy import SharedPolicy as BasePolicy
    else:
        from algos.tf2algos.base.policy import Policy as BasePolicy

    class Off_Policy(BasePolicy):
        def __init__(self,
                     s_dim,
                     visual_sources,
                     visual_resolution,
                     a_dim,
                     is_continuous,
                     **kwargs):
            super().__init__(
                s_dim=s_dim,
                visual_sources=visual_sources,
                visual_resolution=visual_resolution,
                a_dim=a_dim,
                is_continuous=is_continuous,
                **kwargs)
            self.buffer_size = int(kwargs.get('buffer_size', 10000))
            self.use_priority = kwargs.get('use_priority', False)
            self.n_step = kwargs.get('n_step', False)
            self.use_isw = bool(kwargs.get('use_isw', False))
            self.train_times_per_step = int(kwargs.get('train_times_per_step', 1))

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
            self._running_average(s)
            self.data.add(
                s,
                visual_s,
                a,
                r[:, np.newaxis],   # 升维
                s_,
                visual_s_,
                done[:, np.newaxis]  # 升维
            )

        def no_op_store(self, s, visual_s, a, r, s_, visual_s_, done):
            assert isinstance(a, np.ndarray), "no_op_store need action type is np.ndarray"
            assert isinstance(r, np.ndarray), "no_op_store need reward type is np.ndarray"
            assert isinstance(done, np.ndarray), "no_op_store need done type is np.ndarray"
            self._running_average(s)
            self.data.add(
                s,
                visual_s,
                a,
                r[:, np.newaxis],
                s_,
                visual_s_,
                done[:, np.newaxis]
            )

        def get_transitions(self, data_name_list=['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done']):
            '''
            TODO: Annotation
            '''
            data = self.data.sample()   # 经验池取数据
            if not self.is_continuous and 'a' in data_name_list:
                a_idx = data_name_list.index('a')
                data[a_idx] = sth.int2one_hot(data[a_idx].astype(np.int32), self.a_dim)
            if 's' in data_name_list:
                s_idx = data_name_list.index('s')
                data[s_idx] = self.normalize_vector_obs(data[s_idx])
            if 's_' in data_name_list:
                s_idx = data_name_list.index('s_')
                data[s_idx] = self.normalize_vector_obs(data[s_idx])

            return dict([
                [n, d] for n, d in zip(data_name_list, list(map(self.data_convert, data)))
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
            _pre_process = function_dict.get('pre_process_function', lambda *args: args)    # 预处理过程
            _train = function_dict.get('train_function', lambda *args: (None, {}))   # 训练过程
            _update = function_dict.get('update_function', lambda *args: None)  # maybe need update parameters of target networks
            _summary = function_dict.get('summary_dict', {})    # 记录输出到tensorboard的词典
            _sample_data_list = function_dict.get('sample_data_list', ['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done'])  # 需要从经验池提取的经验
            _train_data_list = function_dict.get('train_data_list', ['ss', 'vvss', 'a', 'r', 'done'])  # 需要从经验池提取的经验

            if self.data.is_lg_batch_size:
                # -----------初始化某些变量---------------
                self.intermediate_variable_reset()
                # --------------------------------------

                # --------------------------------------从经验池中获取数据
                data = self.get_transitions(data_name_list=_sample_data_list)  # default: s, visual_s, a, r, s_, visual_s_, done
                # --------------------------------------

                # --------------------------------------如果使用RNN， 就将s和s‘状态进行拼接处理
                if self.use_rnn:
                    data['ss'] = tf.concat([    # [B, T, N], [B, T, N] => [B, T+1, N]
                        data['s'],
                        data['s_'][:, -1:]
                    ], axis=1)
                    data['vvss'] = tf.concat([
                        data['visual_s'],
                        data['visual_s_'][:, -1:]
                    ], axis=1)
                # --------------------------------------如果不使用RNN， 就将s和s‘状态进行堆叠处理
                else:
                    data['ss'] = tf.concat([data['s'], data['s_']], axis=0)  # [B, N] => [2*B, N]
                    data['vvss'] = tf.concat([data['visual_s'], data['visual_s_']], axis=0)
                # --------------------------------------

                # --------------------------------------预处理过程
                data = _pre_process(data)[0]
                # --------------------------------------

                # --------------------------------------好奇心部分
                if self.use_curiosity:
                    # -------------------------------------- 如果使用RNN，那么需要将数据维度从三维转换为二维
                    if self.use_rnn:
                        data['s'] = tf.reshape(data['s'], [-1, data['s'].shape[-1]])    # [B, T, N] => [B*T, N]
                        data['s_'] = tf.reshape(data['s_'], [-1, data['s_'].shape[-1]])
                        data['visual_s'] = tf.reshape(data['visual_s'], [-1, data['visual_s'].shape[-1]])
                        data['visual_s_'] = tf.reshape(data['visual_s_'], [-1, data['visual_s_'].shape[-1]])
                    crsty_r, crsty_loss, crsty_summaries = self.curiosity_model(
                        *self.get_value_from_dict(data_name_list=['s', 'visual_s', 'a', 's_', 'visual_s_'], data_dict=data))
                    data['r'] += crsty_r
                    self.summaries.update(crsty_summaries)
                else:
                    crsty_loss = tf.constant(value=0., dtype=self._data_type)
                # --------------------------------------

                # --------------------------------------优先经验回放部分，获取重要性比例
                if self.use_priority and self.use_isw:
                    _isw = self.data.get_IS_w().reshape(-1, 1)  # [B, ] => [B, 1]
                    _isw = self.data_convert(_isw)
                else:
                    _isw = tf.constant(value=1., dtype=self._data_type)
                # --------------------------------------

                # --------------------------------------获取需要传给train函数的参数
                _training_data = self.get_value_from_dict(data_name_list=_train_data_list, data_dict=data)
                # --------------------------------------

                # --------------------------------------burn in隐状态部分
                if self.use_rnn:
                    cell_state = self.initial_cell_state()
                    if self.burn_in_time_step > 0:
                        _s, _visual_s = self.data.get_burn_in_states()
                        cell_state = self.get_burn_in_feature(_s, _visual_s, cell_state)
                else:
                    cell_state = (None,)
                # --------------------------------------

                # --------------------------------------训练主程序，返回可能用于PER权重更新的TD error，和需要输出tensorboard的信息
                td_error, summaries = _train(_training_data, _isw, crsty_loss, cell_state)
                # --------------------------------------

                # --------------------------------------更新summary
                self.summaries.update(summaries)
                # --------------------------------------

                # --------------------------------------优先经验回放的更新部分
                if self.use_priority:
                    td_error = np.squeeze(td_error.numpy())
                    self.data.update(td_error, self.train_step)
                # --------------------------------------

                # --------------------------------------target网络的更新部分
                _update()
                # --------------------------------------

                # --------------------------------------更新summary
                self.summaries.update(_summary)
                # --------------------------------------

                # --------------------------------------写summary到tensorboard
                self.write_training_summaries(self.global_step, self.summaries)
                # --------------------------------------

    return Off_Policy
