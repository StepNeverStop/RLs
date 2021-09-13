
import math

import torch.nn as nn

from rls.nn.activations import Act_REGISTER, default_act

Vec_REGISTER = {}


class VectorIdentityNetwork(nn.Sequential):

    def __init__(self, in_dim, *args, **kwargs):
        super().__init__()
        self.h_dim = self.in_dim = in_dim
        self.add_module(f'identity', nn.Identity())


class VectorAdaptiveNetwork(nn.Sequential):

    def __init__(self, in_dim, h_dim=16, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        x = math.log2(self.h_dim)
        y = math.log2(self.in_dim)
        l = math.ceil(x) + 1 if math.ceil(x) == math.floor(x) else math.ceil(x)
        r = math.floor(y) if math.ceil(y) == math.floor(y) else math.ceil(y)

        outs = list(map(lambda x: 2**x, range(l, r)[::-1]))
        ins = [self.in_dim] + outs[:-1]

        for i, (_in, _out) in enumerate(zip(ins, outs)):
            self.add_module(f'linear_{i}', nn.Linear(_in, _out))
            self.add_module(f'{default_act}_{i}', Act_REGISTER[default_act]())

        if outs:
            ins = outs[-1]
        else:
            ins = self.in_dim
        self.add_module('linear', nn.Linear(ins, self.h_dim))
        self.add_module(f'{default_act}', Act_REGISTER[default_act]())


Vec_REGISTER['identity'] = VectorIdentityNetwork
Vec_REGISTER['adaptive'] = VectorAdaptiveNetwork
