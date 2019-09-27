import tensorflow as tf

def swish(x):
    return tf.multiply(x, tf.nn.sigmoid(x))

def mish(x):
    return tf.multiply(x, tf.nn.tanh(tf.nn.softplus(x)))