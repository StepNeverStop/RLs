
import math

from torch.nn import (Sequential,
                      Linear)

from rls.nn.activations import (Act_REGISTER,
                                default_act)

Vec_REGISTER = {}


class VectorConcatNetwork:

    def __init__(self, *args, **kwargs):
        assert 'in_dim' in kwargs.keys(), "assert dim in kwargs.keys()"
        self.h_dim = self.in_dim = int(kwargs['in_dim'])
        pass

    def __call__(self, x):
        return x


class VectorAdaptiveNetwork(Sequential):

    def __init__(self, **kwargs):
        super().__init__()
        assert 'in_dim' in kwargs.keys(), "assert dim in kwargs.keys()"
        self.in_dim = int(kwargs['in_dim'])
        self.h_dim = self.out_dim = int(kwargs.get('out_dim', 16))
        x = math.log2(self.out_dim)
        y = math.log2(self.in_dim)
        l = math.ceil(x) + 1 if math.ceil(x) == math.floor(x) else math.ceil(x)
        r = math.floor(y) if math.ceil(y) == math.floor(y) else math.ceil(y)

        outs = list(map(lambda x: 2**x, range(l, r)[::-1]))
        ins = [self.in_dim] + outs[:-1]

        for i, (_in, _out) in enumerate(zip(ins, outs)):
            self.add_module(f'linear_{i}', Linear(_in, _outs))
            self.add_module(f'{default_act}_{i}', Act_REGISTER[default_act]())

        if outs:
            ins = outs[-1]
        else:
            ins = self.in_dim
        self.add_module('linear', Linear(ins, self.out_dim))
        self.add_module(f'{default_act}', Act_REGISTER[default_act]())


Vec_REGISTER['concat'] = VectorConcatNetwork
Vec_REGISTER['adaptive'] = VectorAdaptiveNetwork
