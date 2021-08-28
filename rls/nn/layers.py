#!/usr/bin/env python3
# encoding: utf-8

import math

import torch as t

Layer_REGISTER = {}

Layer_REGISTER['linear'] = t.nn.Linear

# cite from https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/05.noisy_net.ipynb


class NoisyLinear(t.nn.Module):
    """
    Noisy Net: https://arxiv.org/abs/1706.10295
    Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = t.nn.Parameter(t.Tensor(out_features, in_features))
        self.weight_sigma = t.nn.Parameter(
            t.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", t.Tensor(out_features, in_features)
        )

        self.bias_mu = t.nn.Parameter(t.Tensor(out_features))
        self.bias_sigma = t.nn.Parameter(t.Tensor(out_features))
        self.register_buffer("bias_epsilon", t.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        # NOTE: call this after each update
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return t.nn.functional.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> t.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = t.randn(size)

        return x.sign().mul(x.abs().sqrt())


Layer_REGISTER['noisy'] = NoisyLinear
