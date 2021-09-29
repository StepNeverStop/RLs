import torch as th
import torch.nn as nn

from rls.nn.mlps import MLP


class VectorSA2S(nn.Module):
    """
    given s_{t} and a_{t}, predict s_{t+1}.
    """

    def __init__(self,
                 s_dim,
                 a_dim,
                 hidden_units):
        super().__init__()

        self._s_dim = s_dim
        self._a_dim = a_dim
        self._hidden_units = hidden_units

        self._net = MLP(input_dim=s_dim + a_dim,
                        hidden_units=hidden_units,
                        layer='linear',
                        act_fn='tanh',
                        output_shape=s_dim,
                        out_act=None)

    def forward(self, s, a):
        return self._net(th.cat((s, a), -1))
