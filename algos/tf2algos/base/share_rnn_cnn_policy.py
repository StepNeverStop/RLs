import numpy as np
import tensorflow as tf
from .policy import Policy
from rls.networks import VisualNet, ObsRNN
from typing import List


def _split_with_time(state, cell_state=(None,), record_cs=False, s_and_s_=False):
    '''
    TODO: Annotation
    '''
    if s_and_s_:
        state_s, state_s_ = state[:, :-1], state[:, 1:]    # [B, T+1, N] => [B, T, N], [B, T, N]
        state_s = tf.reshape(state_s, [-1, tf.shape(state_s)[-1]])  # [B, T, N] => [B*T, N]
        state_s_ = tf.reshape(state_s_, [-1, tf.shape(state_s_)[-1]])
        if record_cs:
            return state_s, state_s_, cell_state
        else:
            return state_s, state_s_
    else:
        state = tf.reshape(state, [-1, tf.shape(state)[-1]])
        if record_cs:
            return state, cell_state
        else:
            return state


def _split_without_time(state, record_cs=False, s_and_s_=False):
    '''
    TODO: Annotation
    '''
    if s_and_s_:
        state_s, state_s_ = tf.split(state, num_or_size_splits=2, axis=0)
        if record_cs:
            return state_s, state_s_, (None,)
        else:
            return state_s, state_s_
    else:
        if record_cs:
            return state, (None,)
        else:
            return state


class SharedPolicy(Policy):
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
        if self.use_visual:
            self.visual_feature = int(kwargs.get('visual_feature', 128))
            self.visual_net = VisualNet(self.s_dim, self.visual_dim, self.visual_feature)
            self.other_tv += self.visual_net.trainable_variables
            self.feat_dim = self.visual_net.hdim

        if self.use_rnn:
            self.rnn_units = int(kwargs.get('rnn_units', 16))
            self.burn_in_time_step = int(kwargs.get('burn_in_time_step', 20))
            self.train_time_step = int(kwargs.get('train_time_step', 40))
            self.episode_batch_size = int(kwargs.get('episode_batch_size', 32))
            self.rnn_net = ObsRNN(self.feat_dim, self.rnn_units)
            self.other_tv += self.rnn_net.trainable_variables
            self.feat_dim = self.rnn_units

        self.get_feature = tf.function(func=self.generate_get_feature_function(), experimental_relax_shapes=True)
        self.get_burn_in_feature = tf.function(func=self.generate_get_brun_in_feature_function(), experimental_relax_shapes=True)

    def model_recorder(self, kwargs):
        if self.use_visual:
            kwargs.update(visual_net=self.visual_net)
        if self.use_rnn:
            kwargs.update(rnn_net=self.rnn_net)
        super().model_recorder(kwargs)

    def initial_cell_state(self, batch=None):
        if batch is None:
            batch = self.episode_batch_size
        n = 2 if self.rnn_net.rnn_type == 'lstm' else 1
        return tuple(tf.zeros((batch, self.rnn_units), dtype=tf.float32) for _ in range(n))

    def reset(self):
        if self.use_rnn:
            self.cell_state = self.initial_cell_state(self.n_agents)
        else:
            self.cell_state = (None,)

    def get_cell_state(self):
        return self.cell_state

    def set_cell_state(self, cs):
        self.cell_state = cs

    def partial_reset(self, done):
        self._partial_reset_cell_state(index=np.where(done)[0])

    def _partial_reset_cell_state(self, index: List):
        '''
        根据环境的done的index，局部初始化RNN的隐藏状态
        '''
        assert isinstance(index, (list, np.ndarray))
        if self.cell_state[0] is not None and len(index) > 0:
            _arr = np.ones(shape=self.cell_state[0].shape, dtype=np.float32)    # h, c
            _arr[index] = 0.
            self.cell_state = [c * _arr for c in self.cell_state]        # [A, B] * [A, B] => [A, B] 将某行全部替换为0.

    def generate_get_feature_function(self):
        if self.use_visual and self.use_rnn:
            return self._cnn_rnn_get_feature
        else:
            if self.use_visual:
                return self._cnn_get_feature
            elif self.use_rnn:
                return self._rnn_get_feature
            else:
                def _f(s, visual_s, *, cell_state=None, record_cs=False, s_and_s_=False):
                    '''
                    无CNN 和 RNN 的状态特征提取与分割方法
                    '''
                    return _split_without_time(s, record_cs, s_and_s_)
                return _f

    def _cnn_get_feature(self, s, visual_s, *, cell_state=None, record_cs=False, s_and_s_=False):
        '''
        CNN + DNN， 无RNN的 特征提取方法
        '''
        s, visual_s = self.cast(s, visual_s)
        with tf.device(self.device):
            feature = self.visual_net(s, visual_s)
            return _split_without_time(feature, record_cs, s_and_s_)

    def _rnn_get_feature(self, s, visual_s, *, cell_state=None, record_cs=False, s_and_s_=False):
        '''
        RNN + DNN， 无CNN的 特征提取方法
        '''
        s = self.cast(s)[0]    # [A, N] or [B, T+1, N]
        batch_size = tf.shape(s)[0]
        with tf.device(self.device):
            s = tf.reshape(s, [batch_size, -1, tf.shape(s)[-1]])    # [A, N] => [A, 1, N]
            state, cell_state = self.rnn_net(s, *cell_state)  # [B, T, N] => [B, T, N']
            return _split_with_time(state, cell_state, record_cs, s_and_s_)

    def _cnn_rnn_get_feature(self, s, visual_s, *, cell_state=None, record_cs=False, s_and_s_=False):
        '''
        CNN + RNN + DNN, 既有CNN也有RNN的 特征提取方法
        '''
        s, visual_s = self.cast(s, visual_s)    # [A, N] or [B, T+1, N]
        batch_size = tf.shape(s)[0]
        with tf.device(self.device):
            s = tf.reshape(s, [-1, tf.shape(s)[-1]])    # [B, T+1, N] => [B*(T+1), N], [A, N] => [A, N]
            visual_s = tf.reshape(visual_s, [-1, *tf.shape(visual_s)[2:]])
            feature = self.visual_net(s, visual_s)  # [B*(T+1), N]
            feature = tf.reshape(feature, [batch_size, -1, tf.shape(feature)[-1]])  # [B*(T+1), N] => [B, T+1, N]
            state, cell_state = self.rnn_net(feature, *cell_state)
            return _split_with_time(state, cell_state, record_cs, s_and_s_)

    def generate_get_brun_in_feature_function(self):
        if self.use_visual and self.use_rnn:
            return self._cnn_rnn_get_burn_in_feature
        else:
            return self._rnn_get_burn_in_feature

    def _rnn_get_burn_in_feature(self, s, visual_s, cell_state):
        s = self.cast(s)[0]
        with tf.device(self.device):
            _, cell_state = self.rnn_net(s, *cell_state)
            return cell_state

    def _cnn_rnn_get_burn_in_feature(self, s, visual_s):
        s, visual_s = self.cast(s, visual_s)    # [B, T, N]
        batch_size = tf.shape(s)[0]
        with tf.device(self.device):
            s = tf.reshape(s, [-1, tf.shape(s)[-1]])    # [B*T, N]
            visual_s = tf.reshape(visual_s, [-1, tf.shape(visual_s)[-1]])   # [B*T, N]
            feature = self.visual_net(s, visual_s)  # [B*T, N]
            feature = tf.reshape(feature, [batch_size, -1, tf.shape(feature)[-1]])  # [B*T, N] => [B, T, N]
            _, cell_state = self.rnn_net(feature)
            return cell_state
