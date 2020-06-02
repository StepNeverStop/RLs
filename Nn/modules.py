import tensorflow as tf
from tensorflow.keras import Model as M

class DoubleQ(M):
    def __init__(self, q):
        super().__init__()
        self.Q1 = q()
        self.Q2 = q()

    def call(self, *args):
        return self.Q1(*args), self.Q2(*args)

    def get_min(self, *args):
        return tf.minimum(*self(*args))

    def get_max(self, *args):
        return tf.maximum(*self(*args))
