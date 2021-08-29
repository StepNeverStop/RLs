
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch as t
from torch.nn import Identity, Linear, Sequential

from rls.nn.activations import Act_REGISTER, default_act

Rnn_REGISTER = {}


class IdentityRNN(t.nn.Module):

    def __init__(self, in_dim, *args, **kwargs):
        super().__init__()
        self.h_dim = self.in_dim = in_dim

    def forward(self, x, *args, **kwargs):
        return x, None


class GRU_RNN(t.nn.Module):

    def __init__(self, in_dim, rnn_units=16, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.rnn = t.nn.GRUCell(self.in_dim, rnn_units)
        self.h_dim = rnn_units

    def forward(self, feat, cell_state: Optional[Dict], begin_mask: Optional[t.Tensor]):
        '''
        params:
            feat: [T, B, *]
            cell_state: [T, B, *]
        returns:
            output: [T, B, *] or [B, *]
            cell_states: [T, B, *] or [B, *]
        '''
        T, B = feat.shape[:2]

        output = []
        cell_states = defaultdict(list)

        if cell_state:
            hx = cell_state['hx'][0]
        else:
            hx = t.zeros(size=(B, self.h_dim))
        for i in range(T):  # T
            if begin_mask is not None:
                hx *= (1 - begin_mask[i])
            hx = self.rnn(feat[i, ...], hx)

            output.append(hx)
            cell_states['hx'].append(hx)

        output = t.stack(output, dim=0)  # [T, B, N]
        if cell_states:
            cell_states = {k: t.stack(v, 0)
                           for k, v in cell_states.items()}  # [T, B, N]
        return output, cell_states


class LSTM_RNN(t.nn.Module):

    def __init__(self, in_dim, rnn_units=16, **kwargs):
        super().__init__()
        self.in_dim = in_dim
        self.rnn = t.nn.LSTMCell(self.in_dim, rnn_units)
        self.h_dim = rnn_units

    def forward(self, feat, cell_state: Optional[Dict], begin_mask: Optional[t.Tensor]):
        '''
        params:
            feat: [T, B, *]
            cell_state: [T, B, *]
        returns:
            output: [T, B, *] or [B, *]
            cell_states: [T, B, *] or [B, *]
        '''
        T, B = feat.shape[:2]

        output = []
        cell_states = defaultdict(list)

        if cell_state:
            hx, cx = cell_state['hx'][0], cell_state['cx'][0]
        else:
            hx, cx = t.zeros(size=(B, self.h_dim)), t.zeros(
                size=(B, self.h_dim))
        for i in range(T):  # T
            if begin_mask is not None:
                hx *= (1 - begin_mask[i])
                cx *= (1 - begin_mask[i])
            hx, cx = self.rnn(feat[i, ...], (hx, cx))

            output.append(hx)
            cell_states['hx'].append(hx)
            cell_states['cx'].append(cx)

        output = t.stack(output, dim=0)  # [T, B, N]
        if cell_states:
            cell_states = {k: t.stack(v, 0)
                           for k, v in cell_states.items()}  # [T, B, N]

        return output, cell_states


Rnn_REGISTER['identity'] = Rnn_REGISTER['none'] = IdentityRNN
Rnn_REGISTER['gru'] = GRU_RNN
Rnn_REGISTER['lstm'] = LSTM_RNN
