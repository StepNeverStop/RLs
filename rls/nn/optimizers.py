import torch as th

OP_REGISTER = {}
OP_REGISTER['adam'] = th.optim.Adam
OP_REGISTER['sgd'] = th.optim.SGD
OP_REGISTER['rmsprop'] = th.optim.RMSprop
