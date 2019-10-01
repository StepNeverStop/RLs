from .config import *
import tensorflow as tf
try:
    tf_version = tf.version.VERSION[0]
except:
    tf_version = tf.VERSION[0]
finally:
    if tf_version == '1':
        from .tf1algos import *
    elif tf_version == '2':
        from .tf2algos import *