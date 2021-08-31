
import numpy as np
import torch as t
from torch import nn
from torch.nn import functional as F

from rls.nn.activations import Act_REGISTER


class ValueModel(nn.Module):
    def __init__(self, state_dim, rnn_hidden_dim, hidden_dim=400, act='elu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + rnn_hidden_dim, hidden_dim),
            Act_REGISTER[act](),
            nn.Linear(hidden_dim, hidden_dim),
            Act_REGISTER[act](),
            nn.Linear(hidden_dim, hidden_dim),
            Act_REGISTER[act](),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, rnn_hidden):
        return self.net(t.cat([state, rnn_hidden], dim=-1))


class ActionModel(nn.Module):
    def __init__(self, state_dim, rnn_hidden_dim, action_dim, is_continuous,
                 hidden_dim=400, act='elu', min_stddev=5.0, init_stddev=5.0):
        super().__init__()
        self._is_continuous = is_continuous
        self.net = nn.Sequential(
            nn.Linear(state_dim + rnn_hidden_dim, hidden_dim),
            Act_REGISTER[act](),
            nn.Linear(hidden_dim, hidden_dim),
            Act_REGISTER[act](),
            nn.Linear(hidden_dim, hidden_dim),
            Act_REGISTER[act](),
            nn.Linear(hidden_dim, hidden_dim)
        )
        if self._is_continuous:
            self.fc_mean = nn.Linear(hidden_dim, action_dim)
            self.fc_stddev = nn.Linear(hidden_dim, action_dim)
            self.min_stddev = min_stddev
            self.init_stddev = np.log(np.exp(init_stddev) - 1)
        else:
            self.fc_logits = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, rnn_hidden):
        hidden = self.net(t.cat([state, rnn_hidden], dim=-1))   # [B, *]

        if self._is_continuous:
            mean = self.fc_mean(hidden)  # [B, *]
            mean = 5.0 * t.tanh(mean / 5.0)  # [B, *]
            stddev = self.fc_stddev(hidden)  # [B, *]
            stddev = F.softplus(stddev + self.init_stddev) + \
                self.min_stddev    # [B, *]

            return mean, stddev
        else:
            return self.fc_logits(hidden)
