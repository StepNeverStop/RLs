import tensorflow as tf
if tf.version.VERSION[0] == '1':
    from .tf1nn import *
elif tf.version.VERSION[0] == '2':
    from .tf2nn import *
from .noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
