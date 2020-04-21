import numpy as np
import tensorflow as tf
from .base import Base
from abc import abstractmethod
from Nn.networks import CuriosityModel, VisualNet, ObsRNN
from typing import List


class Policy(Base):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,
                 **kwargs):
        super().__init__(**kwargs)
        self.s_dim = s_dim
        self.visual_sources = visual_sources
        if visual_sources >= 1:
            self.use_visual = True
            self.visual_dim = [visual_sources, *visual_resolution]
        else:
            self.use_visual = False
            self.visual_dim = [0]
        self.visual_feature = int(kwargs.get('visual_feature', 128))
        self.visual_net = VisualNet(self.s_dim, self.visual_dim, self.visual_feature)

        self.batch_size = int(kwargs.get('batch_size', 128))

        self.use_rnn = bool(kwargs.get('use_rnn', False))
        self.rnn_units = int(kwargs.get('rnn_units', 16))
        self.burn_in_time_step = int(kwargs.get('burn_in_time_step', 20))
        self.train_time_step = int(kwargs.get('train_time_step', 40))
        self.rnn_net = ObsRNN(self.visual_net.hdim, self.rnn_units, self.use_rnn)
        self.cell_state = None

        self.other_tv = self.visual_net.trainable_variables + self.rnn_net.trainable_variables

        self.is_continuous = is_continuous
        self.a_dim_or_list = a_dim_or_list
        self.gamma = float(kwargs.get('gamma', 0.999))
        self.max_episode = int(kwargs.get('max_episode', 1000))
        self.a_counts = int(np.asarray(a_dim_or_list).prod())
        self.episode = 0    # episode of now

        self.use_curiosity = bool(kwargs.get('use_curiosity', False))
        if self.use_curiosity:
            self.curiosity_eta = float(kwargs.get('curiosity_reward_eta'))
            self.curiosity_lr = float(kwargs.get('curiosity_lr'))
            self.curiosity_beta = float(kwargs.get('curiosity_beta'))
            self.curiosity_loss_weight = float(kwargs.get('curiosity_loss_weight'))
            self.curiosity_model = CuriosityModel(self.is_continuous, self.s_dim, self.a_counts, self.visual_dim, 128, 
                                                  eta=self.curiosity_eta, lr=self.curiosity_lr, beta=self.curiosity_beta, loss_weight=self.curiosity_loss_weight)

        self.get_feature = tf.function(
            func=self.generate_get_feature_function(),
            experimental_relax_shapes=True)
        self.get_burn_in_feature = tf.function(
            func=self.generate_get_brun_in_feature_function(), 
            experimental_relax_shapes=True)

    def reset(self):
        self.cell_state = None

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
        if self.cell_state is not None and len(index) > 0:
            _arr = np.ones(shape=self.cell_state[0].shape, dtype=np.float32)    # h, c
            _arr[index] = 0.
            self.cell_state = [c * _arr for c in self.cell_state]        # [A, B] * [A, B] => [A, B] 将某行全部替换为0.


    def model_recorder(self, kwargs):
        kwargs.update(dict(
            global_step=self.global_step,
            visual_net=self.visual_net,
            rnn_net=self.rnn_net))
        if self.use_curiosity:
            kwargs.update(curiosity_model=self.curiosity_model)
        self.generate_recorder(kwargs)
        self.show_logo()

    def intermediate_variable_reset(self):
        '''
        TODO: Annotation
        '''
        self.summaries = {}

    def get_max_episode(self):
        """
        get the max episode of this training model.
        """
        return self.max_episode

    @abstractmethod
    def choose_action(self, s, visual_s, evaluation=False):
        '''
        choose actions while training.
        Input: 
            s: vector observation
            visual_s: visual observation
        Output: 
            actions
        '''
        pass

    def update_target_net_weights(self, tge, src, ployak=None):
        '''
        update weights of target neural network.
        '''
        if ployak is None:
            tf.group([t.assign(s) for t, s in zip(tge, src)])
        else:
            tf.group([t.assign(ployak * t + (1 - ployak) * s) for t, s in zip(tge, src)])

    @tf.function
    def _get_action(self, s, visual_s, is_training=True):
        '''
        TODO: Annotation
        '''
        raise NotImplementedError

    def set_buffer(self, buffer):
        '''
        TODO: Annotation
        '''
        pass

    def generate_get_feature_function(self):
        # return self._combined_get_features

        if self.use_visual and self.use_rnn:
            return self._cnn_rnn_get_feature
        else:
            if self.use_visual:
                return self._cnn_get_feature
            elif self.use_rnn:
                return self._rnn_get_feature
            else:
                def _f(s, visual_s, *, cell_state=None, record_cs=False, train=True, s_and_s_=False):
                    '''
                    无CNN 和 RNN 的状态特征提取与分割方法
                    '''
                    if s_and_s_:
                        state_s, state_s_ = tf.split(s, num_or_size_splits=2, axis=0)
                        if record_cs:
                            return state_s, state_s_, None
                        else:
                            return state_s, state_s_
                    else:
                        if record_cs:
                            return s, None
                        else:
                            return s
                return _f


    def _cnn_get_feature(self, s, visual_s, *, cell_state=None, record_cs=False, train=True, s_and_s_=False):
        '''
        CNN + DNN， 无RNN的 特征提取方法
        '''
        s, visual_s = self.cast(s, visual_s)
        with tf.device(self.device):
            feature = self.visual_net(s, visual_s)
            if s_and_s_:
                state_s, state_s_ = tf.split(feature, num_or_size_splits=2, axis=0)
                if record_cs:
                    return state_s, state_s_, None
                else:
                    return state_s, state_s_
            else:
                if record_cs:
                    return feature, None
                else:
                    return feature

    def _rnn_get_feature(self, s, visual_s, *, cell_state=None, record_cs=False, train=True, s_and_s_=False):
        '''
        RNN + DNN， 无CNN的 特征提取方法
        '''
        s = self.cast(s)[0]    # [A, N] or [B, T+1, N]
        batch_size = tf.shape(s)[0]
        with tf.device(self.device):
            s = tf.reshape(s, [batch_size, -1, tf.shape(s)[-1]])    # [A, N] => [A, 1, N]
            state, cell_state = self.rnn_net(s, cell_state) # [B, T, N] => [B, T, N']

            if s_and_s_:
                if train:
                    state_s, state_s_ = state[:, :-1], state[:, 1:]    # [B, T+1, N] => [B, T, N], [B, T, N]
                    state_s = tf.reshape(state_s, [-1, tf.shape(state_s)[-1]])  # [B, T, N] => [B*T, N]
                    state_s_ = tf.reshape(state_s_, [-1, tf.shape(state_s_)[-1]])
                else:
                    raise Exception('IF train==False, s_and_s_ must not equal to False.')

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

    def _cnn_rnn_get_feature(self, s, visual_s, *, cell_state=None, record_cs=False, train=True, s_and_s_=False):
        '''
        CNN + RNN + DNN, 既有CNN也有RNN的 特征提取方法
        '''
        s, visual_s = self.cast(s, visual_s)    # [A, N] or [B, T+1, N]
        batch_size = tf.shape(s)[0]
        with tf.device(self.device):
            s = tf.reshape(s, [-1, tf.shape(s)[-1]])    # [B, T+1, N] => [B*(T+1), N], [A, N] => [A, N]
            visual_s = tf.reshape(visual_s, [-1, tf.shape(visual_s)[-1]])
            feature = self.visual_net(s, visual_s)  # [B*(T+1), N]
            feature = tf.reshape(feature, [batch_size, -1, tf.shape(feature)[-1]])  # [B*(T+1), N] => [B, T+1, N]
            state, cell_state = self.rnn_net(feature, cell_state)

            if s_and_s_:
                if train:
                    state_s, state_s_ = state[:, :-1], state[:, 1:]    # [B, T+1, N] => [B, T, N], [B, T, N]
                    state_s = tf.reshape(state_s, [-1, tf.shape(state_s)[-1]])  # [B, T, N] => [B*T, N]
                    state_s_ = tf.reshape(state_s_, [-1, tf.shape(state_s_)[-1]])
                else:
                    raise Exception('IF train==False, s_and_s_ must not equal to False.')

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

    def _combined_get_features(self, s, visual_s, *, cell_state=None, record_cs=False, train=True, s_and_s_=False):
        '''
        CNN or RNN or DNN, 综合功能的特征提取方法，CNN 和 RNN 可以任意搭配、有无
        一般不用
        '''
        s, visual_s = self.cast(s, visual_s)    # [A, N] or [B, T+1, N]
        batch_size = tf.shape(s)[0]
        with tf.device(self.device):
            s = tf.reshape(s, [-1, tf.shape(s)[-1]])    # [B, T+1, N] => [B*(T+1), N], [A, N] => [A, N]
            visual_s = tf.reshape(visual_s, [-1, tf.shape(visual_s)[-1]])
            feature = self.visual_net(s, visual_s)  # [B*(T+1), N]
            feature = tf.reshape(feature, [batch_size, -1, tf.shape(feature)[-1]])  # [B*(T+1), N] => [B, T+1, N]
            state, cell_state = self.rnn_net(feature, cell_state)

            if s_and_s_:
                if train:
                    if self.use_rnn:
                        state_s, state_s_ = state[:, :-1], state[:, 1:]    # [B, T+1, N] => [B, T, N], [B, T, N]
                        state_s = tf.reshape(state_s, [-1, tf.shape(state_s)[-1]])  # [B, T, N] => [B*T, N]
                        state_s_ = tf.reshape(state_s_, [-1, tf.shape(state_s_)[-1]])
                    else:
                        state = tf.reshape(state, [batch_size, -1])
                        state_s, state_s_ = tf.split(state, num_or_size_splits=2, axis=0)
                else:
                    raise Exception('IF train==False, s_and_s_ must not equal to False.')

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

    def generate_get_brun_in_feature_function(self):
        if self.use_visual and self.use_rnn:
            return self._cnn_rnn_get_burn_in_feature
        else:
            return self._rnn_get_burn_in_feature

    def _rnn_get_burn_in_feature(self, s, visual_s):
        s = self.cast(s)[0]
        with tf.device(self.device):
            _, cell_state = self.rnn_net(s)
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
