import tensorflow as tf
if tf.__version__[0] == '1':
    from .tf1nn import *
elif tf.__version__[0] == '2':
    from .tf2nn import *
from .noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
