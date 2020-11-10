#!/usr/bin/env python3
# encoding: utf-8

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras import Model as M
from tensorflow.keras import Input as I
from tensorflow.keras.layers import (Conv2D,
                                     Dense,
                                     Flatten)

from rls.utils.tf2_utils import get_device
from rls.utils.indexs import VisualEncoderType
from rls.nn.layers import ConvLayer
from rls.nn.activations import default_activation
from rls.nn.initializers import initKernelAndBias


def get_visual_encoder_from_type(encoder_type: VisualEncoderType):
    VISUAL_ENCODER_FUNCS = {
        VisualEncoderType.SIMPLE: lambda: ConvLayer(Conv2D, [16, 32], [[8, 8], [4, 4]], [[4, 4], [2, 2]], padding='valid', activation='elu'),
        VisualEncoderType.NATURE: lambda: ConvLayer(Conv2D, [32, 64, 64], [[8, 8], [4, 4], [3, 3]], [[4, 4], [2, 2], [1, 1]], padding='valid', activation='relu'),
        VisualEncoderType.MATCH3: lambda: ConvLayer(Conv2D, [35, 144], [[3, 3], [1, 1]], [[3, 3], [1, 1]], padding='valid', activation='elu')
    }
    return VISUAL_ENCODER_FUNCS.get(encoder_type, VISUAL_ENCODER_FUNCS[VisualEncoderType.SIMPLE])


class MultiCameraCNN(M):
    '''多个图像来源输入的CNN，未初始化
    '''

    def __init__(self, n, feature_dim, activation_fn, encoder_type):
        super().__init__()
        self.n = n
        self.nets = []
        for _ in range(n):
            net = get_visual_encoder_from_type(encoder_type)
            net.add(Dense(feature_dim, activation_fn, **initKernelAndBias))
            self.nets.append(net)

    def call(self, vector_input, visual_input):
        f = [self.nets[i](visual_input[:, i]) for i in range(self.n)]
        f = tf.concat([vector_input, *f], axis=-1)
        return f


class ObsLSTM(M):
    '''输入状态的RNN
    '''

    def __init__(self, dim, hidden_units):
        super().__init__()
        self.rnn_type = 'lstm'
        # self.masking = tf.keras.layers.Masking(mask_value=0.)

        # ValueError: Tried to convert 'tensor' to a tensor and failed. Error: None values not supported.
        # https://github.com/tensorflow/tensorflow/issues/31998
        cell = tf.keras.layers.LSTMCell(hidden_units)
        self.lstm_net = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True)
        self(I(shape=(None, dim)), I(shape=(hidden_units,)), I(shape=(hidden_units,)))

    def call(self, s, h, c):
        # s = self.masking(s)
        x, h, c = self.lstm_net(s, initial_state=(h, c))  # 如果没指定初始化隐状态，就用burn_in的， 或者 None
        return (x, (h, c))


class ObsGRU(M):
    '''输入状态的RNN
    '''

    def __init__(self, dim, hidden_units):
        super().__init__()
        self.rnn_type = 'gru'
        cell = tf.keras.layers.GRUCell(hidden_units)
        self.lstm_net = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True)
        self(I(shape=(None, dim)), I(shape=(hidden_units,)))

    def call(self, s, h):
        x, h = self.lstm_net(s, initial_state=h)  # 如果没指定初始化隐状态，就用burn_in的， 或者 None
        return (x, (h,))


class VisualNet(M):
    '''
    Processing image input observation information.
    The feature obtained by forward propagation will be concatenate with the vector input.
    If there is no visual image input, Conv layers won't be built and initialized.
    '''

    def __init__(self, vector_dim, visual_dim=[], visual_feature=128, encoder_type=VisualEncoderType.NATURE):
        super().__init__()
        self.camera_num = visual_dim[0]
        self.nets = MultiCameraCNN(n=self.camera_num, feature_dim=visual_feature, activation_fn=default_activation, encoder_type=encoder_type)
        self.hdim = vector_dim + (visual_feature * self.camera_num) * (self.camera_num > 0)
        self(I(shape=vector_dim), I(shape=visual_dim))

    def call(self, vector_input, visual_input):
        visual_input = tf.cast(visual_input, tf.float32)
        return self.nets(vector_input, visual_input)


class CuriosityModel(M):
    '''
    Model of Intrinsic Curiosity Module (ICM).
    Curiosity-driven Exploration by Self-supervised Prediction, https://arxiv.org/abs/1705.05363
    '''

    def __init__(self, is_continuous, vector_dim, action_dim, visual_dim=[], visual_feature=128,
                 *, eta=0.2, lr=1.0e-3, beta=0.2, loss_weight=10., encoder_type=VisualEncoderType.SIMPLE):
        '''
        params:
            is_continuous: sepecify whether action space is continuous(True) or discrete(False)
            vector_dim: dimension of vector state input
            action_dim: dimension of action
            visual_dim: dimension of visual state input
            visual_feature: dimension of visual feature map
            eta: weight of intrinsic reward
            lr: the learning rate of curiosity model
            beta: weight factor of loss between inverse_dynamic_net and forward_net
            loss_weight: weight factor of loss between policy gradient and curiosity model
        '''
        super().__init__()
        self.device = get_device()
        self.eta = eta
        self.beta = beta
        self.loss_weight = loss_weight
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.is_continuous = is_continuous

        self.camera_num = visual_dim[0]
        if self.camera_num == 0:
            self.use_visual = False
        else:
            self.use_visual = True

        self.nets = MultiCameraCNN(n=self.camera_num, feature_dim=visual_feature, activation_fn=default_activation, encoder_type=encoder_type)
        self.s_dim = vector_dim + (visual_feature * self.camera_num) * (self.camera_num > 0)

        if self.use_visual:
            # S, S' => A
            self.inverse_dynamic_net = Sequential([
                Dense(self.s_dim * 2, default_activation, **initKernelAndBias),
                Dense(action_dim, 'tanh' if is_continuous else None, **initKernelAndBias)
            ])

        # S, A => S'
        self.forward_net = Sequential([
            Dense(self.s_dim + action_dim, default_activation, **initKernelAndBias),
            Dense(self.s_dim, None, **initKernelAndBias)
        ])
        self.initial_weights(I(shape=vector_dim), I(shape=visual_dim), I(shape=action_dim))

        self.tv = []
        if self.use_visual:
            for net in self.nets:
                self.tv += net.trainable_variables
            self.tv += self.inverse_dynamic_net.trainable_variables
        self.tv += self.forward_net.trainable_variables

    def initial_weights(self, vector_input, visual_input, action):
        s = self.nets(vector_input, visual_input)
        if self.use_visual:
            self.inverse_dynamic_net(tf.concat((s, s), -1))
        self.forward_net(tf.concat((s, action), -1))

    @tf.function(experimental_relax_shapes=True)
    def call(self, s, visual_s, a, s_, visual_s_):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                fs = [self.nets[i](visual_s[:, i]) for i in range(self.camera_num)]
                fs_ = [self.nets[i](visual_s_[:, i]) for i in range(self.camera_num)]

                fsa = tf.concat((*fs, s, a), axis=-1)            # <S, A>
                s_target = tf.concat((*fs_, s_), axis=-1)        # S'
                s_eval = self.forward_net(fsa)                  # <S, A> => S'
                LF = 0.5 * tf.reduce_sum(tf.square(s_target - s_eval), axis=-1, keepdims=True)    # [B, 1]
                intrinsic_reward = self.eta * LF
                loss_forward = tf.reduce_mean(LF)

                if self.use_visual:
                    f = tf.concat((*fs, s, *fs_, s_), axis=-1)
                    a_eval = self.inverse_dynamic_net(f)
                    if self.is_continuous:
                        loss_inverse = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(a_eval - a), axis=-1))
                    else:
                        idx = tf.argmax(a, axis=-1)  # [B, ]
                        loss_inverse = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=idx, logits=a_eval))
                    loss = (1 - self.beta) * loss_inverse + self.beta * loss_forward
                else:
                    loss = loss_forward

            grads = tape.gradient(loss, self.tv)
            self.optimizer.apply_gradients(zip(grads, self.tv))
            summaries = dict([
                ['LOSS/curiosity_loss', loss],
                ['LOSS/forward_loss', loss_forward]
            ])
            if self.use_visual:
                summaries.update({
                    'LOSS/inverse_loss': loss_inverse
                })
            # crsty_loss = loss * self.loss_weight
            return intrinsic_reward, summaries
