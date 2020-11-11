#!/usr/bin/env python3
# encoding: utf-8

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.keras.layers import (Dense,
                                     GaussianNoise,
                                     Conv2D,
                                     Flatten)

from rls.nn.activations import default_activation
from rls.nn.initializers import initKernelAndBias


class mlp(Sequential):
    def __init__(self, hidden_units, *, layer=Dense, act_fn=default_activation, output_shape=1, out_activation=None, out_layer=True):
        """
        Args:
            hidden_units: like [32, 32]
            output_shape: units of last layer
            out_activation: activation function of last layer
            out_layer: whether need specifing last layer or not
        """
        super().__init__()
        for u in hidden_units:
            self.add(layer(u, act_fn, **initKernelAndBias))
        if out_layer:
            self.add(layer(output_shape, out_activation, **initKernelAndBias))


class mlp_with_noisy(Sequential):
    def __init__(self, hidden_units, act_fn=default_activation, output_shape=1, out_activation=None, out_layer=True):
        """
        Add a gaussian noise to to the result of Dense layer. The added gaussian noise is not related to the origin input.
        Args:
            hidden_units: like [32, 32]
            output_shape: units of last layer
            out_activation: activation function of last layer
            out_layer: whether need specifing last layer or not
        """
        super().__init__()
        for u in hidden_units:
            self.add(GaussianNoise(0.4))  # Or use kwargs
            self.add(Dense(u, act_fn, **initKernelAndBias))
        if out_layer:
            self.add(GaussianNoise(0.4))
            self.add(Dense(output_shape, out_activation, **initKernelAndBias))


class Noisy(Dense):
    '''
    Noisy Net: https://arxiv.org/abs/1706.10295
    Add the result of another noisy net to the result of origin Dense layer.
    '''

    def __init__(self, units, activation=None, **kwargs):
        super().__init__(units, activation=None, **kwargs)
        self.noise_sigma = float(kwargs.get('noise_sigma', .4))
        self.mode = str(kwargs.get('noisy_distribution', 'independent'))  # independent or factorised
        self.noisy_activation = activations.get(activation)

    def build(self, input_shape):
        super().build(input_shape)
        self.built = False
        self.last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.noisy_w = self.add_weight(
            'noise_kernel',
            shape=[self.last_dim, self.units],
            initializer=tf.random_normal_initializer(0.0, .1),
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.noisy_b = self.add_weight(
                'noise_bias',
                shape=[self.units, ],
                initializer=tf.constant_initializer(self.noise_sigma / (self.units**0.5)),
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None
        self.built = True

    def funcForFactor(self, x):
        return tf.sign(x) * tf.pow(tf.abs(x), 0.5)

    @property
    def epsilon_w(self):
        if self.mode == 'independent':
            return tf.random.truncated_normal([self.last_dim, self.units], stddev=self.noise_sigma)
        elif self.mode == 'factorised':
            return self.funcForFactor(tf.random.truncated_normal([self.last_dim, 1], stddev=self.noise_sigma)) \
                * self.funcForFactor(tf.random.truncated_normal([1, self.units], stddev=self.noise_sigma))

    @property
    def epsilon_b(self):
        return tf.random.truncated_normal([self.units, ], stddev=self.noise_sigma)

    def noisy_layer(self, inputs):
        return tf.matmul(inputs, self.noisy_w * self.epsilon_w) + self.noisy_b * self.epsilon_b

    def call(self, inputs, need_noise=True):
        y = super().call(inputs)
        if need_noise:
            noise = self.noisy_layer(inputs)
            outputs = y + noise
        else:
            outputs = y

        if self.noisy_activation is not None:
            return self.noisy_activation(outputs)
        else:
            return outputs


def ConvLayer(conv_function=Conv2D,
              filters=[32, 64, 64],
              kernels=[[8, 8], [4, 4], [3, 3]],
              strides=[[4, 4], [2, 2], [1, 1]],
              padding='valid',
              activation='relu'):
    '''
    Params:
        conv_function: the convolution function
        filters: list of flitter of all hidden conv layers
        kernels: list of kernel of all hidden conv layers
        strides: list of stride of all hidden conv layers
        padding: padding mode
        activation: activation function
    Return:
        A sequential of multi-convolution layers, with Flatten.
    '''
    layers = Sequential([conv_function(filters=f, kernel_size=k, strides=s, padding=padding, activation=activation, **initKernelAndBias) for f, k, s in zip(filters, kernels, strides)])
    layers.add(Flatten())
    return layers
