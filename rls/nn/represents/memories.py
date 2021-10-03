from collections import defaultdict
from typing import Dict, Optional

import torch as th
import torch.nn as nn

Rnn_REGISTER = {}


class IdentityRNN(nn.Module):

    def __init__(self, in_dim, *args, **kwargs):
        super().__init__()
        self.h_dim = self.in_dim = in_dim

    def forward(self, x, *args, **kwargs):
        return x, None


class GRU_RNN(nn.Module):

    def __init__(self, in_dim, rnn_units=16, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.rnn = nn.GRUCell(self.in_dim, rnn_units)
        self.h_dim = rnn_units

    def forward(self, feat, rnncs: Optional[Dict], begin_mask: Optional[th.Tensor]):
        """
        params:
            feat: [T, B, *]
            rnncs: [T, B, *]
        returns:
            output: [T, B, *] or [B, *]
            rnncs_s: [T, B, *] or [B, *]
        """
        T, B = feat.shape[:2]

        output = []
        rnncs_s = defaultdict(list)

        if rnncs:
            hx = rnncs['hx'][0]
        else:
            hx = th.zeros(size=(B, self.h_dim))
        for i in range(T):  # T
            if begin_mask is not None:
                hx *= (1 - begin_mask[i])
            hx = self.rnn(feat[i, ...], hx)

            output.append(hx)
            rnncs_s['hx'].append(hx)

        output = th.stack(output, dim=0)  # [T, B, N]
        if rnncs_s:
            rnncs_s = {k: th.stack(v, 0)
                       for k, v in rnncs_s.items()}  # [T, B, N]
        return output, rnncs_s


class LSTM_RNN(nn.Module):

    def __init__(self, in_dim, rnn_units=16, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.rnn = nn.LSTMCell(self.in_dim, rnn_units)
        self.h_dim = rnn_units

    def forward(self, feat, rnncs: Optional[Dict], begin_mask: Optional[th.Tensor]):
        """
        params:
            feat: [T, B, *]
            rnncs: [T, B, *]
        returns:
            output: [T, B, *] or [B, *]
            rnncs_s: [T, B, *] or [B, *]
        """
        T, B = feat.shape[:2]

        output = []
        rnncs_s = defaultdict(list)

        if rnncs:
            hx, cx = rnncs['hx'][0], rnncs['cx'][0]
        else:
            hx, cx = th.zeros(size=(B, self.h_dim)), th.zeros(
                size=(B, self.h_dim))
        for i in range(T):  # T
            if begin_mask is not None:
                hx *= (1 - begin_mask[i])
                cx *= (1 - begin_mask[i])
            hx, cx = self.rnn(feat[i, ...], (hx, cx))

            output.append(hx)
            rnncs_s['hx'].append(hx)
            rnncs_s['cx'].append(cx)

        output = th.stack(output, dim=0)  # [T, B, N]
        if rnncs_s:
            rnncs_s = {k: th.stack(v, 0)
                       for k, v in rnncs_s.items()}  # [T, B, N]

        return output, rnncs_s


Rnn_REGISTER['identity'] = Rnn_REGISTER['none'] = IdentityRNN
Rnn_REGISTER['gru'] = GRU_RNN
Rnn_REGISTER['lstm'] = LSTM_RNN
