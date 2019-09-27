from .config import *
import tensorflow as tf
if tf.version.VERSION[0] == '1':
    from .tf1algos import *
elif tf.version.VERSION[0] == '2':
    from .tf2algos import *