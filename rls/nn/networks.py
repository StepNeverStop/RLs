#!/usr/bin/env python3
# encoding: utf-8

import torch as t
import numpy as np

from typing import (Tuple,
                    Dict,
                    Optional)
from torch.nn import (Sequential,
                      Linear)
from collections import defaultdict

from rls.nn.activations import (default_act,
                                Act_REGISTER)

from rls.nn.represents.vectors import Vec_REGISTER
from rls.nn.represents.visuals import Vis_REGISTER


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
        # h, w, c => c, h, w
        visual_inputs = [vi.swapaxes(-1, -3).swapaxes(-1, -2)
                         for vi in visual_inputs]
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
            self.rnn = t.nn.GRUCell(feat_dim, rnn_units)
        elif self.network_type == 'lstm':
            self.rnn = t.nn.LSTMCell(feat_dim, rnn_units)

    def forward(self, feat, cell_state: Optional[Dict]):
        '''
        params:
            feat: [T, B, *]
            cell_state: [T, B, *]
        returns:
            output: [T, B, *] or [B, *]
            cell_states: [T, B, *] or [B, *]
        '''

        T = feat.shape[0]

        output = []
        cell_states = defaultdict(list)
        if self.network_type == 'gru':
            if cell_state:
                hx = cell_state['hx'][0]
            else:
                hx = None
            for i in range(T):  # T
                hx = self.rnn(feat[i, ...], hx)

                output.append(hx)
                cell_states['hx'].append(hx)

        elif self.network_type == 'lstm':
            if cell_state:
                hc = cell_state['hx'][0], cell_state['cx'][0]
            else:
                hc = None
            for i in range(T):  # T
                hx, cx = self.rnn(feat[i, ...], hc)
                hc = (hx, cx)

                output.append(hx)
                cell_states['hx'].append(hx)
                cell_states['cx'].append(cx)
        if T > 1:
            output = t.stack(output, dim=0)  # [T, B, N]
            cell_states = {k: t.stack(v, 0)
                           for k, v in cell_states.items()}  # [T, B, N]
            return output, cell_states
        else:
            # [B, *]
            return output[0], {k: v[0] for k, v in cell_states.items()}
