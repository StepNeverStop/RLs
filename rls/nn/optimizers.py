
import torch as t
from torch.optim import Optimizer

OP_REGISTER = {}
OP_REGISTER['adam'] = t.optim.Adam
OP_REGISTER['sgd'] = t.optim.SGD
OP_REGISTER['rmsprop'] = t.optim.RMSprop
