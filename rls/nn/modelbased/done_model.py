import torch as th
import torch.distributions as td
import torch.nn as nn

from rls.nn.mlps import MLP


class VectorSA2D(nn.Module):
    """
    given s_{t} and a_{t}, predict done_{t+1}.
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
                        output_shape=1,
                        out_act=None)

    def forward(self, s, a):
        logits = self._net(th.cat((s, a), -1))
        return td.independent.Independent(td.Bernoulli(logits=logits, validate_args=False), 1)
