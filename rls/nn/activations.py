#!/usr/bin/env python3
# encoding: utf-8
import torch as th
import torch.nn as nn

default_act = 'relu'
Act_REGISTER = {}

Act_REGISTER[None] = lambda: lambda x: x
Act_REGISTER['relu'] = nn.ReLU
Act_REGISTER['elu'] = nn.ELU
Act_REGISTER['gelu'] = nn.GELU
Act_REGISTER['leakyrelu'] = nn.LeakyReLU
Act_REGISTER['tanh'] = nn.Tanh
Act_REGISTER['softplus'] = nn.Softplus
Act_REGISTER['mish'] = nn.Mish
Act_REGISTER['sigmoid'] = nn.Sigmoid
Act_REGISTER['log_softmax'] = lambda: nn.LogSoftmax(-1)


class Swish(nn.Module):
    """
    https://arxiv.org/abs/1710.05941
    """

    def forward(self, inp: th.Tensor) -> th.Tensor:
        return inp * th.sigmoid(inp)


Act_REGISTER['swish'] = Swish
