import torch as t
import torch.nn as nn
import torch.nn.functional as F

from rls.nn.mlps import MLP
from rls.nn.represent_nets import RepresentationNetwork

# Vanilla Variational Auto-Encoder


class VAE(nn.Module):

    def __init__(self, obs_spec, rep_net_params,
                 a_dim, z_dim, hiddens=dict()):
        super().__init__()

        self.z_dim = z_dim

        self.rep_net = RepresentationNetwork(obs_spec=obs_spec,
                                             rep_net_params=rep_net_params)

        self._encoder = MLP(input_dim=self.rep_net.h_dim + a_dim,
                            hidden_units=hiddens['encoder'],
                            act_fn='relu',
                            output_shape=z_dim*2)

        self._decoder = MLP(input_dim=self.rep_net.h_dim + z_dim,
                            hidden_units=hiddens['decoder'],
                            act_fn='relu',
                            output_shape=a_dim,
                            out_act='tanh')

    def forward(self, x, a, **kwargs):
        x, _ = self.rep_net(x, **kwargs)
        encoder_output = self._encoder(t.cat([x, a], -1))
        mean, log_std = t.chunk(encoder_output, 2, -1)
        log_std = log_std.clamp(-4, 15)
        std = t.exp(log_std)
        z = mean + std * t.randn_like(std)

        u = self._decoder(t.cat([x, z], -1))

        return u, mean, std

    def decode(self, x, z=None, **kwargs):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        x, _ = self.rep_net(x, **kwargs)
        if z is None:
            z = t.randn(x.shape[:-1]+(self.z_dim,)).clamp(-0.5, 0.5)
        return self._decoder(t.cat([x, z], -1))
