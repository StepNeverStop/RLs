#!/usr/bin/env python3
# encoding: utf-8

import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects

# from rls.utils.specs import DefaultActivationFuncType

class Mish(Activation):

    def __init__(self, activation, **kwargs):
        super().__init__(activation, **kwargs)
        self.__name__ = 'mish'


def mish(x):
    """
    Swish activation function. For more info: https://arxiv.org/abs/1908.08681
    The original repository for Mish: https://github.com/digantamisra98/Mish
    """
    return tf.multiply(x, tf.nn.tanh(tf.nn.softplus(x)))

get_custom_objects().update({
    'mish': Mish(mish)
    })


default_activation = 'swish'  # 'tanh', 'relu', 'swish', 'mish'
