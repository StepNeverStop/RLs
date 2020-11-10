#!/usr/bin/env python3
# encoding: utf-8

import tensorflow as tf

# from rls.utils.indexs import DefaultActivationFuncType


def swish(x):
    """Swish activation function. For more info: https://arxiv.org/abs/1710.05941"""
    return tf.multiply(x, tf.nn.sigmoid(x))


def mish(x):
    """
    Swish activation function. For more info: https://arxiv.org/abs/1908.08681
    The original repository for Mish: https://github.com/digantamisra98/Mish
    """
    return tf.multiply(x, tf.nn.tanh(tf.nn.softplus(x)))


default_activation = 'relu'  # 'tanh', 'relu', swish, mish
