import tensorflow as tf


def swish(x):
    """Swish activation function. For more info: https://arxiv.org/abs/1710.05941"""
    return tf.multiply(x, tf.nn.sigmoid(x))


def mish(x):
    """
    Swish activation function. For more info: https://arxiv.org/abs/1908.08681
    The original repository for Mish: https://github.com/digantamisra98/Mish
    """
    return tf.multiply(x, tf.nn.tanh(tf.nn.softplus(x)))
