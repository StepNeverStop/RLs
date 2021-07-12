#!/usr/bin/env python3
# encoding: utf-8

import uuid
import torch as t

from typing import Tuple

from torch.nn import (Sequential,
                      Linear)

from rls.nn.activations import (default_act,
                                Act_REGISTER)

from rls.nn.vector_nets import Vec_REGISTER
from rls.nn.visual_nets import Vis_REGISTER


class MultiVectorNetwork(t.nn.Module):
    def __init__(self, vector_dim=[], network_type='concat'):
        super().__init__()
        self.nets = t.nn.ModuleList()
        for in_dim in vector_dim:
            self.nets.append(Vec_REGISTER[network_type](in_dim=in_dim))
        self.h_dim = sum([net.h_dim for net in self.nets])

    def forward(self, *vector_inputs):
        output = []
        for net, s in zip(self.nets, vector_inputs):
            output.append(net(s))
        output = t.cat(output, -1)
        return output


class MultiVisualNetwork(t.nn.Module):

    def __init__(self, visual_dim=[], visual_feature=128, network_type='nature'):
        super().__init__()
        self.dense_nets = t.nn.ModuleList()
        for vd in visual_dim:
            net = Vis_REGISTER[network_type](visual_dim=vd)
            self.dense_nets.append(
                Sequential(
                    net,
                    Linear(net.output_dim, visual_feature),
                    Act_REGISTER[default_act]()
                )
            )
        self.h_dim = visual_feature * len(self.dense_nets)

    def forward(self, *visual_inputs):
        visual_inputs = [vi.permute(0, 3, 1, 2) for vi in visual_inputs]
        output = []
        for dense_net, visual_s in zip(self.dense_nets, visual_inputs):
            output.append(
                dense_net(visual_s)
            )
        output = t.cat(output, -1)
        return output


class EncoderNetwork(t.nn.Module):
    def __init__(self, feat_dim=64, output_dim=64):
        super().__init__()
        self.h_dim = output_dim
        self.net = Linear(feat_dim, output_dim)
        self.act = Act_REGISTER[default_act]()

    def forward(self, feat):
        return self.act(self.net(feat))


class MemoryNetwork(t.nn.Module):
    def __init__(self, feat_dim=64, rnn_units=8, *, network_type='lstm'):
        super().__init__()
        self.h_dim = rnn_units
        self.network_type = network_type

        if self.network_type == 'gru':
            self.cell_nums = 1
            self.rnn = t.nn.GRUCell(feat_dim, rnn_units)
        elif self.network_type == 'lstm':
            self.cell_nums = 2
            self.rnn = t.nn.LSTMCell(feat_dim, rnn_units)

    def forward(self, feat, cell_state=None):
        # feat: [B, T, x]
        output = []
        for i in range(feat.shape[1]):
            cell_state = self.rnn(feat[:, i], cell_state)
            if self.network_type == 'gru':
                output.append(cell_state)
            elif self.network_type == 'lstm':
                output.append(cell_state[0])
        output = t.stack(output, dim=0)  # [T, B, N]
        output.permute(1, 0, 2)  # [B, T, N]
        return output, cell_state

    def initial_cell_state(self, batch: int) -> Tuple[t.Tensor]:
        return tuple(t.zeros((batch, self.h_dim)).float() for _ in range(self.cell_nums))
