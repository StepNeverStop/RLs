import numpy as np
import torch as t
from torch import distributions as td
from torch import nn
from torch.nn import functional as F

from rls.nn.activations import Act_REGISTER
from rls.nn.dreamer.distributions import SampleDist, TanhBijector


class VisualDecoder(nn.Module):
    """
    p(o_t | s_t, h_t)
    Observation model to reconstruct image observation (3, 64, 64)
    from state and rnn hidden state
    """

    def __init__(self, feat_dim,
                 visual_dim, depth=32, act='relu'):
        super().__init__()
        self.visual_dim = visual_dim
        self.fc = nn.Linear(feat_dim, 32*depth)
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
        #     _hidden = self.fc(t.zeros(1, feat_dim))
        #     _hidden = _hidden.view(1, 1024, 1, 1)    # [B, 1024, 1, 1]
        #     self.h_dim = self.net(_hidden).shape

    def forward(self, feat):
        '''
        feat: [T, B, *] or [B, *]
        '''
        tb = feat.shape[:-1]
        hidden = self.fc(feat)  # [B, 1024]
        hidden = hidden.view(t.prod(tb), 1024, 1, 1)    # [B, 1024, 1, 1]
        obs = self.net(hidden)  # [B, c, h, w]
        obs = obs.view(tb+obs[-3:])
        obs = obs.swapaxes(-2, -3).swapaxes(-1, -2)   # [B, h, w, c]
        obs_dist = td.Independent(td.Normal(obs, 1), len(self.visual_dim))
        return obs_dist


class VectorDecoder(nn.Module):
    """
    """

    def __init__(self, feat_dim, vector_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, vector_dim)
        )
        self.h_dim = vector_dim

    def forward(self, feat):
        obs = self.net(feat)   # [B, *] or [T, B, *]
        obs_dist = td.Independent(td.Normal(obs, 1), 1)
        return obs_dist


class DenseModel(nn.Module):
    def __init__(self, feature_size: int, output_shape: tuple, layers: int, hidden_size: int, dist='normal',
                 activation=nn.ELU):
        super().__init__()
        self._output_shape = output_shape
        self._layers = layers
        self._hidden_size = hidden_size
        self._dist = dist
        self.activation = activation
        # For adjusting pytorch to tensorflow
        self._feature_size = feature_size
        # Defining the structure of the NN
        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for i in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self._hidden_size,
                            int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, features):
        dist_inputs = self.model(features)
        reshaped_inputs = t.reshape(
            dist_inputs, features.shape[:-1] + self._output_shape)
        if self._dist == 'normal':
            return td.independent.Independent(td.Normal(reshaped_inputs, 1), len(self._output_shape))
        if self._dist == 'binary':
            return td.independent.Independent(td.Bernoulli(logits=reshaped_inputs, validate_args=False), len(self._output_shape))
        raise NotImplementedError(self._dist)


class ActionDecoder(nn.Module):
    def __init__(self, action_size, feature_size, layers, hidden_size, dist='tanh_normal',
                 activation=nn.ELU, min_std=1e-4, init_std=5, mean_scale=5):
        super().__init__()
        self.action_size = action_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.dist = dist
        self.activation = activation
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.feedforward_model = self.build_model()
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)

    def build_model(self):
        model = [nn.Linear(self.feature_size, self.hidden_size)]
        model += [self.activation()]
        for i in range(1, self.layers):
            model += [nn.Linear(self.hidden_size, self.hidden_size)]
            model += [self.activation()]
        if self.dist == 'tanh_normal':
            model += [nn.Linear(self.hidden_size, self.action_size * 2)]
        elif self.dist == 'one_hot' or self.dist == 'relaxed_one_hot':
            model += [nn.Linear(self.hidden_size, self.action_size)]
        else:
            raise NotImplementedError(f'{self.dist} not implemented')
        return nn.Sequential(*model)

    def forward(self, state_features, is_train=True):
        x = self.feedforward_model(state_features)
        dist = None
        if self.dist == 'tanh_normal':
            mean, std = t.chunk(x, 2, -1)
            mean = self.mean_scale * t.tanh(mean / self.mean_scale)
            std = F.softplus(std + self.raw_init_std) + self.min_std
            dist = td.Normal(mean, std)
            dist = td.TransformedDistribution(dist, TanhBijector())
            dist = td.Independent(dist, 1)
            dist = SampleDist(dist)
        elif self.dist == 'one_hot':
            dist = td.OneHotCategorical(logits=x)
        elif self.dist == 'relaxed_one_hot':
            dist = td.RelaxedOneHotCategorical(0.1, logits=x)

        if self.dist == 'tanh_normal':
            if is_train:
                action = dist.rsample()
            else:
                action = dist.mode()
        elif self.dist == 'one_hot':
            action = dist.sample()
            # This doesn't change the value, but gives us straight-through gradients
            action = action + dist.probs - dist.probs.detach()
        elif self.dist == 'relaxed_one_hot':
            action = dist.rsample()
        else:
            action = dist.sample()
        return action
