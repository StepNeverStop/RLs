#!/usr/bin/env python3
# encoding: utf-8
import torch as t

default_act = 'relu'
Act_REGISTER = {}

Act_REGISTER[None] = lambda: lambda x: x
Act_REGISTER['relu'] = t.nn.ReLU
Act_REGISTER['elu'] = t.nn.ELU
Act_REGISTER['gelu'] = t.nn.GELU
Act_REGISTER['leakyrelu'] = t.nn.LeakyReLU
Act_REGISTER['tanh'] = t.nn.Tanh
Act_REGISTER['softplus'] = t.nn.Softplus
Act_REGISTER['mish'] = t.nn.Mish
Act_REGISTER['sigmoid'] = t.nn.Sigmoid
Act_REGISTER['log_softmax'] = t.nn.LogSoftmax


class Swish(t.nn.Module):
    '''
    https://arxiv.org/abs/1710.05941
    '''

    def forward(self, input: t.Tensor) -> t.Tensor:
        return input * t.sigmoid(input)


Act_REGISTER['swish'] = Swish
