
import numpy as np
import torch as t
from torch import distributions as td
from torch import nn
from torch.nn import functional as F

from rls.nn.activations import Act_REGISTER
from rls.nn.dreamer.distributions import OneHotDistFlattenSample
from rls.nn.layers import Layer_REGISTER


class RecurrentStateSpaceModel(nn.Module):
    """
    This class includes multiple components
    Deterministic stoch_state model: h_t+1 = f(h_t, s_t, a_t)
    Stochastic stoch_state model (prior): p(s_t+1 | h_t+1)
    State posterior: q(s_t | h_t, o_t)
    NOTE: actually, this class takes embedded observation by Encoder class
    min_stddev is added to stddev same as original implementation
    Activation function for this class is F.relu same as original implementation
    """

    def __init__(self, stoch_dim, deter_dim, action_dim, obs_embed_dim,
                 hidden_units=200, discretes=0, min_stddev=0.1, act='relu',
                 layer='linear', std_act='softplus'):
        super().__init__()

        self.stoch_dim = stoch_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self._discretes = discretes
        self._std_act = std_act
        self._act = act
        self._layer = layer

        self._rssm_type = 'discrete' if self._discretes > 0 else 'continuous'
        if self._rssm_type == 'discrete':
            self._input_dim = self.stoch_dim * self._discretes
            output_dim = self.stoch_dim * self._discretes
        else:
            self._input_dim = self.stoch_dim
            output_dim = self.stoch_dim * 2

        self.fc_state_action = nn.Sequential(
            Layer_REGISTER[self._layer](
                self._input_dim + action_dim, hidden_units),
            Act_REGISTER[self._act]()
        )
        self.rnn = nn.GRUCell(hidden_units, deter_dim)
        self.fc_output_prior = nn.Sequential(
            Layer_REGISTER[self._layer](deter_dim, hidden_units),
            Act_REGISTER[self._act](),
            Layer_REGISTER[self._layer](hidden_units, output_dim)
        )
        self.fc_output_posterior = nn.Sequential(
            Layer_REGISTER[self._layer](deter_dim + obs_embed_dim, hidden_units),
            Act_REGISTER[self._act](),
            Layer_REGISTER[self._layer](hidden_units, output_dim)
        )

        self._min_stddev = min_stddev

    def forward(self, stoch_state, action, deter_state, embedded_next_obs):
        """
        h_t+1 = f(h_t, s_t, a_t)
        Return prior p(s_t+1 | h_t+1) and posterior p(s_t+1 | h_t+1, o_t+1)
        for model training
        """
        next_state_prior, deter_state = self.prior(
            stoch_state, action, deter_state)     # [B, *]
        next_state_posterior = self.posterior(
            deter_state, embedded_next_obs)
        return next_state_prior, next_state_posterior, deter_state

    def prior(self, stoch_state, action, deter_state):
        """
        h_t+1 = f(h_t, s_t, a_t)
        Compute prior p(s_t+1 | h_t+1)
        """
        hidden = self.fc_state_action(t.cat([stoch_state, action], dim=-1))  # [B, *]
        deter_state = self.rnn(hidden, deter_state)    # [B, *]
        output = self.fc_output_prior(deter_state)  # [B, *]
        return self._build_dist(output), deter_state

    def posterior(self, deter_state, embedded_obs):
        """
        Compute posterior q(s_t | h_t, o_t)
        """
        output = self.fc_output_posterior(
            t.cat([deter_state, embedded_obs], dim=-1))
        return self._build_dist(output)

    def _build_dist(self, output):
        if self._rssm_type == 'discrete':
            logits = output.view(
                output.shape[:-1]+(self.stoch_dim, self._discretes))   # [B, s, d]
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

    def clone_dist(self, dist, detach=False):
        if self._rssm_type == 'discrete':
            mean = dist.mean
            if detach:
                mean = t.detach(mean)
            return td.Independent(OneHotDistFlattenSample(mean), 1)
        else:
            mean, stddev = dist.mean, dist.stddev
            if detach:
                mean, stddev = t.detach(mean), t.detach(stddev)
            return td.Independent(td.Normal(mean, stddev), 1)

    def init_state(self, shape):
        if not hasattr(shape, '__len__'):
            shape = (shape,)
        return t.zeros(shape+(self._input_dim,)), t.zeros(shape+(self.deter_dim,))
