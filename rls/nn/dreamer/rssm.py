
import numpy as np
import torch as t
from torch import distributions as td
from torch import nn
from torch.nn import functional as F

from rls.nn.dreamer.distributions import OneHotDistFlattenSample


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
                 hidden_dim=200, discretes=0, min_stddev=0.1, act=F.elu,
                 std_act='softplus'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self._discretes = discretes
        self._std_act = std_act

        if self._discretes > 0:
            input_dim = self.state_dim * self._discretes
            output_dim = self.state_dim * self._discretes
        else:
            input_dim = self.state_dim
            output_dim = self.state_dim * 2

        self.fc_state_action = nn.Linear(input_dim + action_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, rnn_hidden_dim)
        self.fc_rnn_hidden = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.fc_output_prior = nn.Linear(hidden_dim, output_dim)

        self.fc_rnn_hidden_embedded_obs = nn.Linear(
            rnn_hidden_dim + obs_embed_dim, hidden_dim)
        self.fc_output_posterior = nn.Linear(hidden_dim, output_dim)

        self._min_stddev = min_stddev
        self.act = act

    def forward(self, state, action, rnn_hidden, embedded_next_obs, build_dist=True):
        """
        h_t+1 = f(h_t, s_t, a_t)
        Return prior p(s_t+1 | h_t+1) and posterior p(s_t+1 | h_t+1, o_t+1)
        for model training
        """
        next_state_prior, rnn_hidden = self.prior(
            state, action, rnn_hidden, build_dist)     # [B, *]
        next_state_posterior = self.posterior(
            rnn_hidden, embedded_next_obs, build_dist)
        return next_state_prior, next_state_posterior, rnn_hidden

    def prior(self, state, action, rnn_hidden, build_dist=True):
        """
        h_t+1 = f(h_t, s_t, a_t)
        Compute prior p(s_t+1 | h_t+1)
        """
        hidden = self.act(self.fc_state_action(
            t.cat([state, action], dim=-1)))  # [B, *]
        rnn_hidden = self.rnn(hidden, rnn_hidden)    # [B, *]
        hidden = self.act(self.fc_rnn_hidden(rnn_hidden))    # [B, *]
        output = self.fc_output_prior(hidden)  # [B, *]
        if build_dist:
            output = self._build_dist(output)
        return output, rnn_hidden

    def posterior(self, rnn_hidden, embedded_obs, build_dist=True):
        """
        Compute posterior q(s_t | h_t, o_t)
        """
        hidden = self.act(self.fc_rnn_hidden_embedded_obs(
            t.cat([rnn_hidden, embedded_obs], dim=-1)))  # [B, *]
        output = self.fc_output_posterior(hidden)  # [B, *]
        if build_dist:
            output = self._build_dist(output)
        return output

    def _build_dist(self, output):
        if self._discretes > 0:
            logits = output.view(
                output.shape[:-1]+(self.state_dim, self._discretes))   # [B, s, d]
            return td.Independent(OneHotDistFlattenSample(logits=logits), 1)
        else:
            mean, stddev = t.chunk(output, 2, -1)   # [B, *]
            if self._std_act == 'softplus':
                stddev = F.softplus(stddev)
            elif self._std_act == 'sigmoid':
                stddev = t.sigmoid(stddev)
            elif self._std_act == 'sigmoid2':
                stddev = 2. * t.sigmoid(stddev / 2.)

            stddev = stddev + self._min_stddev  # [B, *]
            return td.Independent(td.Normal(mean, stddev), 1)
