# https://github.com/juliusfrost/dreamer-pytorch/blob/master/dreamer/models/distribution.py

import numpy as np
import torch as t
import torch.nn.functional as F
from torch import distributions as td
from torch.distributions import constraints


class SampleDist:

    def __init__(self, dist: td.Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    @property
    def mean(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        return t.mean(sample, 0)

    @property
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


class OneHotDist(td.OneHotCategorical):

    # NOTE: td.OneHotCategoricalStraightThrough will act wrongly when using with td.Independent

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def rsample(self, *args, **kwargs):
        # Straight through biased gradient estimator.
        samples = super().sample(*args, **kwargs)
        # This doesn't change the value, but gives us straight-through gradients
        samples = samples + self.probs - self.probs.detach()
        return samples


class OneHotDistFlattenSample(OneHotDist):
    # TODO: check

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample(self, *args, **kwargs):
        samples = super().sample(*args, **kwargs)
        assert samples.ndim > 1, 'assert samples.ndim > 1'
        samples = samples.view(samples.shape[:-2]+(-1,))
        return samples

    def rsample(self, *args, **kwargs):
        samples = super().rsample(*args, **kwargs)
        assert samples.ndim > 1, 'assert samples.ndim > 1'
        samples = samples.view(samples.shape[:-2]+(-1,))
        return samples
