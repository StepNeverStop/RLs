#!/usr/bin/env python3
# encoding: utf-8

from typing import Dict, Optional

import torch as th
import torch.nn as nn

from rls.nn.activations import Act_REGISTER, default_act
from rls.nn.represents.encoders import End_REGISTER
from rls.nn.represents.memories import Rnn_REGISTER
from rls.nn.represents.vectors import Vec_REGISTER
from rls.nn.represents.visuals import Vis_REGISTER


class MultiVectorNetwork(nn.Module):
    def __init__(self, vector_dim=[], h_dim=16, network_type='identity'):
        super().__init__()
        self.nets = nn.ModuleList()
        for in_dim in vector_dim:
            self.nets.append(Vec_REGISTER[network_type](
                in_dim=in_dim, h_dim=h_dim))
        self.h_dim = sum([net.h_dim for net in self.nets])

    def forward(self, *vector_inputs):
        output = []
        for net, s in zip(self.nets, vector_inputs):
            output.append(net(s))
        output = th.cat(output, -1)
        return output


class MultiVisualNetwork(nn.Module):

    def __init__(self, visual_dim=[], h_dim=128, network_type='nature'):
        super().__init__()
        self.dense_nets = nn.ModuleList()
        for vd in visual_dim:
            net = Vis_REGISTER[network_type](visual_dim=vd)
            self.dense_nets.append(
                nn.Sequential(
                    net,
                    nn.Linear(net.output_dim, h_dim),
                    Act_REGISTER[default_act]()
                )
            )
        self.h_dim = h_dim * len(self.dense_nets)

    def forward(self, *visual_inputs):
        # h, w, c => c, h, w
        batch = visual_inputs[0].shape[:-3]
        visual_inputs = [vi.view((-1,) + vi.shape[-3:]).swapaxes(-1, -3).swapaxes(-1, -2)
                         for vi in visual_inputs]
        output = []
        for dense_net, visual_s in zip(self.dense_nets, visual_inputs):
            output.append(
                dense_net(visual_s).view(batch + (-1,))
            )
        output = th.cat(output, -1)
        return output


class EncoderNetwork(nn.Module):
    def __init__(self, feat_dim=64, h_dim=64, network_type='identity'):
        super().__init__()
        self.net = End_REGISTER[network_type](in_dim=feat_dim, h_dim=h_dim)
        self.h_dim = self.net.h_dim

    def forward(self, feat):
        return self.net(feat)


class MemoryNetwork(nn.Module):
    def __init__(self, feat_dim=64, rnn_units=8, network_type='lstm'):
        super().__init__()
        self.net = Rnn_REGISTER[network_type](in_dim=feat_dim,
                                              rnn_units=rnn_units)
        self.h_dim = self.net.h_dim

    def forward(self, feat, rnncs: Optional[Dict], begin_mask: Optional[th.Tensor]):
        """
        params:
            feat: [T, B, *]
            rnncs: [T, B, *]
        returns:
            output: [T, B, *] or [B, *]
            rnncs_s: [T, B, *] or [B, *]
        """

        _squeeze = False
        if feat.ndim == 2:  # [B, *]
            _squeeze = True
            feat = feat.unsqueeze(0)  # [B, *] => [1, B, *]
            if rnncs:
                rnncs = {k: v.unsqueeze(0)  # [1, B, *]
                         for k, v in rnncs.items()}

        output, rnncs_s = self.net(feat, rnncs, begin_mask)  # [B, *] or [T, B, *]

        if _squeeze:
            output = output.squeeze(0)  # [B, *]
            if rnncs_s:
                rnncs_s = {k: v.squeeze(0)
                           for k, v in rnncs_s.items()}  # [B, *]
        return output, rnncs_s
