#!/usr/bin/env python3
# encoding: utf-8

import tensorflow as tf

from typing import Tuple
from tensorflow.keras import Model as M
from tensorflow.keras import Input as I
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Conv2D,
                                     MaxPool2D,
                                     AveragePooling2D,
                                     Flatten,
                                     Dense,
                                     BatchNormalization)

from rls.nn.layers import ConvLayer
from rls.nn.activations import default_activation
from rls.nn.initializers import initKernelAndBias
from rls.utils.specs import (VisualNetworkType,
                             MemoryNetworkType)


def get_visual_network_from_type(network_type: VisualNetworkType):
    VISUAL_NETWORKS = {
        VisualNetworkType.SIMPLE: lambda: ConvLayer(Conv2D, [16, 32], [[8, 8], [4, 4]], [[4, 4], [2, 2]], padding='valid', activation='elu'),
        VisualNetworkType.NATURE: lambda: ConvLayer(Conv2D, [32, 64, 64], [[8, 8], [4, 4], [3, 3]], [[4, 4], [2, 2], [1, 1]], padding='valid', activation='relu'),
        VisualNetworkType.MATCH3: lambda: ConvLayer(Conv2D, [35, 144], [[3, 3], [1, 1]], [[3, 3], [1, 1]], padding='valid', activation='elu'),
        VisualNetworkType.RESNET: ResnetNetwork,
        VisualNetworkType.DEEPCONV: DeepConvNetwork
    }
    return VISUAL_NETWORKS.get(network_type, VISUAL_NETWORKS[VisualNetworkType.SIMPLE])


class DeepConvNetwork(Sequential):

    def __init__(self,
                 filters=[16, 32],
                 kernel_sizes=[[8, 8], [4, 4]],
                 strides=[[4, 4], [2, 2]],
                 padding='valid',

                 use_bn=False,

                 max_pooling=False,
                 avg_pooling=False,
                 pool_sizes=[[2, 2], [2, 2]],
                 pool_strides=[[1, 1], [1, 1]],
                 ):
        super().__init__()
        for i in range(conv_layers):
            self.add(Conv2D(filters=filters[i],
                            kernel_size=kernel_sizes[i],
                            strides=strides[i],
                            padding=padding,
                            activation='relu',
                            **initKernelAndBias))
            if use_bn:
                self.add(BatchNormalization())

            if max_pooling:
                self.add(MaxPool2D(pool_size=pool_sizes[i],
                                   strides=pool_strides[i]))
            elif avg_pooling:
                self.add(AveragePooling2D(pool_size=pool_sizes[i],
                                          strides=pool_strides[i]))
        self.add(Flatten())


class ResnetNetwork(M):

    def __init__(self):
        super().__init__()
        self.all_filters = [16, 32, 32]
        self.res_blocks = 2
        for i, filters in enumerate(self.all_filters):
            setattr(self, 'conv' + str(i), Conv2D(filters=filters, kernel_size=[3, 3], strides=(1, 1), **initKernelAndBias))
            setattr(self, 'pool' + str(i), MaxPool2D(pool_size=[3, 3], strides=[2, 2], padding='same'))
            for j in range(self.res_blocks):
                setattr(self, 'resblock' + str(i) + 'conv' + str(j), Conv2D(filters=filters, kernel_size=[3, 3], strides=(1, 1), padding='same', **initKernelAndBias))
        self.flatten = Flatten()

    def call(self, x):
        '''
           -----------------------------------multi conv layer---------------------------------
           ↓                                             ----multi residual block-------      ↑
           ↓                                             ↓                             ↑      ↑
        x - > conv -> x -> max_pooling -> x(block_x) -> relu -> x -> resnet_conv -> x => x ↘ ↑
                                               ↓                                         +    x -> relu -> x -> flatten -> x
                                               --------------residual add----------------↑ ↗
        '''
        for i in range(len(self.all_filters)):
            x = getattr(self, 'conv' + str(i))(x)
            block_x = x = getattr(self, 'pool' + str(i))(x)
            for j in range(self.res_blocks):
                x = tf.nn.relu(x)
                x = getattr(self, 'resblock' + str(i) + 'conv' + str(j))(x)
            x = tf.add(block_x, x)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        return x


class MultiVectorNetwork(M):
    def __init__(self, vector_dim=[]):
        # TODO
        super().__init__()
        self.nets = []
        for _ in vector_dim:
            def net(x): return x
            self.nets.append(net)
        self.h_dim = sum(vector_dim)
        self.use_vector = not self.h_dim == 0
        if vector_dim:
            self(*(I(shape=dim) for dim in vector_dim))

    @tf.function
    def call(self, *args):
        output = []
        for net, s in zip(self.nets, args):
            output.append(net(s))
        if output:
            output = tf.concat(output, axis=-1)
        return output


class MultiVisualNetwork(M):

    def __init__(self, visual_dim=[], visual_feature=128, network_type=VisualNetworkType.NATURE):
        super().__init__()
        self.nets = []
        self.dense_nets = []
        for _ in visual_dim:
            net = get_visual_network_from_type(network_type)()
            self.nets.append(net)
            self.dense_nets.append(Dense(visual_feature, default_activation, **initKernelAndBias))
        self.h_dim = visual_feature * len(self.nets)
        self.use_visual = not self.h_dim == 0
        if visual_dim:
            self(*(I(shape=dim) for dim in visual_dim))

    @tf.function
    def call(self, *args):
        output = []
        for net, dense_net, visual_s in zip(self.nets, self.dense_nets, args):
            output.append(
                dense_net(
                    net(visual_s)
                )
            )
        if output:
            output = tf.concat(output, axis=-1)
        return output


class EncoderNetwork(M):
    def __init__(self, feat_dim=64, output_dim=64, *, use_encoder=False):
        # TODO
        super().__init__()
        self.use_encoder = use_encoder
        self.h_dim = output_dim if use_encoder else feat_dim
        self.net = Dense(output_dim, default_activation, **initKernelAndBias) if use_encoder else lambda x: x
        self(I(shape=feat_dim))

    @tf.function
    def call(self, feat):
        return self.net(feat)


class MemoryNetwork(M):
    def __init__(self, feat_dim=64, rnn_units=8, *, use_rnn=False, network_type=MemoryNetworkType.LSTM):
        super().__init__()
        # self.masking = tf.keras.layers.Masking(mask_value=0.)

        # ValueError: Tried to convert 'tensor' to a tensor and failed. Error: None values not supported.
        # https://github.com/tensorflow/tensorflow/issues/31998
        self.use_rnn = use_rnn
        self.h_dim = rnn_units if use_rnn else feat_dim
        self.network_type = network_type
        if use_rnn:
            if self.network_type == MemoryNetworkType.GRU:
                self.cell_nums = 1
                cell = tf.keras.layers.GRUCell(rnn_units)
            elif self.network_type == MemoryNetworkType.LSTM:
                self.cell_nums = 2
                cell = tf.keras.layers.LSTMCell(rnn_units)
            self.rnn_net = tf.keras.layers.RNN(cell, return_state=True, return_sequences=True)
            self(*(
                [I(shape=(None, feat_dim))]
                +
                [I(shape=rnn_units) for _ in range(self.cell_nums)]
            ))
        else:
            self.cell_nums = 1
            self.rnn_net = lambda x, initial_state: (x, initial_state)

    @tf.function
    def call(self, *args):
        # s = self.masking(s)
        output = self.rnn_net(args[0], initial_state=args[1:] if len(args) > 2 else args[1])
        x = output[0]
        cell_state = output[1:]
        return x, cell_state

    def initial_cell_state(self, batch: int) -> Tuple[tf.Tensor]:
        if self.use_rnn:
            return tuple(tf.zeros((batch, self.h_dim), dtype=tf.float32) for _ in range(self.cell_nums))
        return (None,)
