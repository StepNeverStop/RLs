import torch as t
import torch.nn as nn

from rls.nn.activations import Act_REGISTER, default_act
from rls.nn.layers import Layer_REGISTER

Mlp_REGISTER = {}


class MLP(nn.Sequential):
    '''Multi-Layer Perceptron'''

    def __init__(self,
                 input_dim,
                 hidden_units,
                 *,
                 layer='linear',
                 act_fn=default_act,
                 output_shape=None,
                 out_act=None):
        """
        Args:
            hidden_units: like [32, 32]
            output_shape: units of last layer
            out_act: activation function of last layer
        """
        super().__init__()
        ins = [input_dim] + hidden_units[:-1]
        outs = hidden_units
        for i, (_in, _out) in enumerate(zip(ins, outs)):
            self.add_module(f'{layer}_{i}', Layer_REGISTER[layer](_in, _out))
            self.add_module(f'{act_fn}_{i}', Act_REGISTER[act_fn]())

        input_dim = outs[-1] if len(outs) > 0 else input_dim
        if output_shape:
            self.add_module('out_layer', Layer_REGISTER[layer](
                input_dim, output_shape))
            if out_act:
                self.add_module('out_act', Act_REGISTER[out_act]())


Mlp_REGISTER['mlp'] = MLP
