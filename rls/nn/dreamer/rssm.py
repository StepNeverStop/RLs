
import numpy as np
import torch as t
from torch import distributions as td
from torch import nn
from torch.nn import functional as F


class RecurrentStateSpaceModel(nn.Module):
    """
    This class includes multiple components
    Deterministic state model: h_t+1 = f(h_t, s_t, a_t)
    Stochastic state model (prior): p(s_t+1 | h_t+1)
    State posterior: q(s_t | h_t, o_t)
    NOTE: actually, this class takes embedded observation by Encoder class
    min_stddev is added to stddev same as original implementation
    Activation function for this class is F.relu same as original implementation
    """

    def __init__(self, state_dim, rnn_hidden_dim, action_dim, obs_embed_dim,
                 hidden_dim=200, min_stddev=0.1, act=F.elu):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim

        self.fc_state_action = nn.Linear(state_dim + action_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self.fc_rnn_hidden = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc_state_mean_prior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_prior = nn.Linear(hidden_dim, state_dim)

        self.fc_rnn_hidden_embedded_obs = nn.Linear(
            rnn_hidden_dim + obs_embed_dim, hidden_dim)
        self.fc_state_mean_posterior = nn.Linear(hidden_dim, state_dim)
        self.fc_state_stddev_posterior = nn.Linear(hidden_dim, state_dim)

        self._min_stddev = min_stddev
        self.act = act

    def forward(self, state, action, rnn_hidden, embedded_next_obs):
        """
        h_t+1 = f(h_t, s_t, a_t)
        Return prior p(s_t+1 | h_t+1) and posterior p(s_t+1 | h_t+1, o_t+1)
        for model training
        """
        next_state_prior, rnn_hidden = self.prior(
            state, action, rnn_hidden)     # [B, *]
        next_state_posterior = self.posterior(rnn_hidden, embedded_next_obs)
        return next_state_prior, next_state_posterior, rnn_hidden

    def prior(self, state, action, rnn_hidden):
        """
        h_t+1 = f(h_t, s_t, a_t)
        Compute prior p(s_t+1 | h_t+1)
        """
        hidden = self.act(self.fc_state_action(
            t.cat([state, action], dim=-1)))  # [B, *]
        rnn_hidden = self.rnn(hidden, rnn_hidden)    # [B, *]
        hidden = self.act(self.fc_rnn_hidden(rnn_hidden))    # [B, *]
        mean = self.fc_state_mean_prior(hidden)  # [B, *]
        stddev = F.softplus(self.fc_state_stddev_prior(  # [B, *]
            hidden)) + self._min_stddev
        return td.Independent(td.Normal(mean, stddev), 1), rnn_hidden

    def posterior(self, rnn_hidden, embedded_obs):
        """
        Compute posterior q(s_t | h_t, o_t)
        """
        hidden = self.act(self.fc_rnn_hidden_embedded_obs(
            t.cat([rnn_hidden, embedded_obs], dim=-1)))  # [B, *]
        mean = self.fc_state_mean_posterior(hidden)  # [B, *]
        stddev = F.softplus(self.fc_state_stddev_posterior(
            hidden)) + self._min_stddev  # [B, *]
        return td.Independent(td.Normal(mean, stddev), 1)
