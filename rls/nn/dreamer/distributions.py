# https://github.com/juliusfrost/dreamer-pytorch/blob/master/dreamer/models/distribution.py

import numpy as np
import torch as t
import torch.nn.functional as F
from torch import distributions as td
from torch.distributions import constraints


class TanhBijector(td.Transform):

    def __init__(self):
        super().__init__()
        # https://discuss.pytorch.org/t/solved-pytorch1-8-attributeerror-tanhbijector-object-has-no-attribute-domain/116092/6
        self.bijective = True
        self.domain = constraints.Constraint()
        self.codomain = constraints.Constraint()

    @property
    def sign(self):
        return 1.

    def _call(self, x):
        return t.tanh(x)

    def _inverse(self, y: t.Tensor):
        y = t.where(
            (t.abs(y) <= 1.),
            t.clamp(y, -0.99999997, 0.99999997),
            y
        )

        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2. * (np.log(2) - x - F.softplus(-2. * x))


class SampleDist:

    def __init__(self, dist: td.Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        return t.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = t.argmax(logprob, dim=0).reshape(
            1, batch_size, 1).expand(1, batch_size, feature_size)
        return t.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -t.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()


def atanh(x):
    return 0.5 * t.log((1 + x) / (1 - x))
