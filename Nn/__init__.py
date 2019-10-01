import tensorflow as tf
try:
    tf_version = tf.version.VERSION[0]
except:
    tf_version = tf.VERSION[0]
finally:
    if tf_version == '1':
        from .tf1nn import *
    elif tf_version == '2':
        from .tf2nn import *
from .noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
