import numpy as np
import torch as t
from torch import nn
from torch.nn import functional as F

from rls.nn.activations import Act_REGISTER


class VisualDecoder(nn.Module):
    """
    p(o_t | s_t, h_t)
    Observation model to reconstruct image observation (3, 64, 64)
    from state and rnn hidden state
    """

    def __init__(self, state_dim, rnn_hidden_dim,
                 visual_dim, depth=32, act='relu'):
        super().__init__()
        self.fc = nn.Linear(state_dim + rnn_hidden_dim, 32*depth)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(32*depth, 4*depth, kernel_size=5, stride=2),
            Act_REGISTER[act](),
            nn.ConvTranspose2d(4*depth, 2*depth, kernel_size=5, stride=2),
            Act_REGISTER[act](),
            nn.ConvTranspose2d(2*depth, 1*depth, kernel_size=6, stride=2),
            Act_REGISTER[act](),
            nn.ConvTranspose2d(
                1*depth, visual_dim[-1], kernel_size=6, stride=2)
        )
        # with t.no_grad():
        #     _hidden = self.fc(t.zeros(1, state_dim + rnn_hidden_dim))
        #     _hidden = _hidden.view(1, 1024, 1, 1)    # [B, 1024, 1, 1]
        #     self.h_dim = self.net(_hidden).shape

    def forward(self, state, rnn_hidden):
        hidden = self.fc(t.cat([state, rnn_hidden], dim=-1))  # [B, 1024]
        hidden = hidden.view(hidden.size(0), 1024, 1, 1)    # [B, 1024, 1, 1]
        obs = self.net(hidden)  # [B, c, h, w]
        obs = obs.permute(0, 2, 3, 1)   # [B, h, w, c]
        return obs


class RewardModel(nn.Module):
    """
    p(r_t | s_t, h_t)
    Reward model to predict reward from state and rnn hidden state
    """

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


class VectorDecoder(nn.Module):
    """
    """

    def __init__(self, state_dim, rnn_hidden_dim, vector_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + rnn_hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, vector_dim),
            nn.ReLU()
        )
        self.h_dim = vector_dim

    def forward(self, state, rnn_hidden):
        return self.net(t.cat([state, rnn_hidden], dim=-1))
