from typing import Union

import numpy as np
import torch as t
import torch.nn.functional as F
from torch import distributions as td
from torch import nn

from rls.nn.activations import Act_REGISTER
from rls.nn.dreamer.distributions import SampleDist
from rls.nn.layers import Layer_REGISTER


class VisualDecoder(nn.Module):
    """
    p(o_t | s_t, h_t)
    Observation model to reconstruct image observation (3, 64, 64)
    from state and rnn hidden state
    """

    def __init__(self, feat_dim, visual_dim,
                 depth=32, act='relu'):
        super().__init__()
        self._depth = depth
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

    def forward(self, feat):
        '''
        feat: [T, B, *] or [B, *]
        '''
        tb = feat.shape[:-1]
        hidden = self.fc(feat)  # [T, B, 32*depth]
        hidden = hidden.view(t.prod(tb), 32*self._depth, 1, 1)    # [T*B, 32*depth, 1, 1]
        obs = self.net(hidden)  # [T*B, c, h, w]
        obs = obs.view(tb+obs[-3:])  # [T, B, c, h, w]
        obs = obs.swapaxes(-2, -3).swapaxes(-1, -2)   # [T, B, h, w, c]
        obs_dist = td.Independent(td.Normal(obs, 1), len(self.visual_dim))
        return obs_dist


class DenseModel(nn.Module):

    def __init__(self,
                 feature_size: int,
                 output_shape: Union[tuple, int, list],
                 layers: int = 1,
                 hidden_units: int = 64,
                 dist='none',
                 activation='relu',
                 layer='linear'):
        super().__init__()
        self._output_shape = (output_shape,) if isinstance(output_shape, int) else output_shape
        self._layers = layers
        self._hidden_units = hidden_units
        self._dist = dist
        self._activation = activation
        self._layer = layer
        # For adjusting pytorch to tensorflow
        self._feature_size = feature_size
        # Defining the structure of the NN
        self.model = self.build_model()

    def build_model(self):
        model = [Layer_REGISTER[self._layer](self._feature_size, self._hidden_units),
                 Act_REGISTER[self._activation]()]

        for i in range(self._layers - 1):
            model += [Layer_REGISTER[self._layer](self._hidden_units, self._hidden_units),
                      Act_REGISTER[self._activation]()]

        model += [Layer_REGISTER[self._layer](self._hidden_units,
                                              int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, features):
        dist_inputs = self.model(features)
        reshaped_inputs = t.reshape(
            dist_inputs, features.shape[:-1] + self._output_shape)
        if self._dist == 'mse':
            return td.independent.Independent(td.Normal(reshaped_inputs, 1), len(self._output_shape))
        elif self._dist == 'binary':
            return td.independent.Independent(td.Bernoulli(logits=reshaped_inputs, validate_args=False), len(self._output_shape))
        elif self._dist == 'none':
            return reshaped_inputs
        else:
            raise NotImplementedError(self._dist)


class ActionDecoder(nn.Module):
    def __init__(self, action_size, feature_size, layers, hidden_units,
                 dist='tanh_normal', activation='relu', layer='linear', min_std=1e-4,
                 init_std=5, mean_scale=5):
        super().__init__()
        self.action_size = action_size
        self.feature_size = feature_size
        self.hidden_units = hidden_units
        self.layers = layers
        self.dist = dist
        self._activation = activation
        self._layer = layer
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.feedforward_model = self.build_model()
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)

    def build_model(self):
        model = [Layer_REGISTER[self._layer](self.feature_size, self.hidden_units),
                 Act_REGISTER[self._activation]()]
        for i in range(self.layers-1):
            model += [Layer_REGISTER[self._layer](self.hidden_units, self.hidden_units),
                      Act_REGISTER[self._activation]()]
        if self.dist in ['tanh_normal', 'trunc_normal']:
            model += [Layer_REGISTER[self._layer](self.hidden_units, self.action_size * 2)]
        elif self.dist in ['one_hot', 'relaxed_one_hot']:
            model += [Layer_REGISTER[self._layer](self.hidden_units, self.action_size)]
        else:
            raise NotImplementedError(f'{self.dist} not implemented')
        return nn.Sequential(*model)

    def forward(self, state_features):
        x = self.feedforward_model(state_features)
        if self.dist == 'tanh_normal':
            mean, std = t.chunk(x, 2, -1)
            mean = self.mean_scale * t.tanh(mean / self.mean_scale)
            std = F.softplus(std + self.raw_init_std) + self.min_std
            dist = td.Normal(mean, std)
            # TODO: fix nan problem
            dist = td.TransformedDistribution(
                dist, td.TanhTransform(cache_size=1))
            dist = td.Independent(dist, 1)
            dist = SampleDist(dist)
        elif self.dist == 'trunc_normal':
            mean, std = t.chunk(x, 2, -1)
            std = 2 * t.sigmoid((std + self.raw_init_std) / 2) + self.min_std
            from rls.nn.dists.TruncatedNormal import \
                TruncatedNormal as TruncNormalDist
            dist = TruncNormalDist(t.tanh(mean), std, -1, 1)
            dist = td.Independent(dist, 1)
        elif self.dist == 'one_hot':
            dist = td.OneHotCategoricalStraightThrough(logits=x)
        elif self.dist == 'relaxed_one_hot':
            dist = td.RelaxedOneHotCategorical(t.tensor(0.1), logits=x)
        return dist

    def sample_actions(self, state_features, is_train=True):
        dist = self(state_features)
        if self.dist in ['tanh_normal', 'trunc_normal']:
            if is_train:
                action = dist.rsample()
            else:
                action = dist.mean
        elif self.dist == 'one_hot':
            action = dist.rsample()
        elif self.dist == 'relaxed_one_hot':
            action = dist.rsample()
        else:
            action = dist.sample()
        return action
