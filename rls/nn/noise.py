#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
from abc import ABC, abstractmethod
# copy from openai baseline https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py


class AdaptiveParamNoiseSpec(object):
    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adoption_coefficient = adoption_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adoption_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adoption_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adoption_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adoption_coefficient)


class ActionNoise(ABC, object):

    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    @abstractmethod
    def __call__(self, size):
        raise NotImplementedError


class NormalActionNoise(ActionNoise):
    def __init__(self, mu=0.0, sigma=1.0):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, size):
        return np.random.normal(self.mu, self.sigma, size)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class ClippedNormalActionNoise(NormalActionNoise):
    def __init__(self, mu=0.0, sigma=1.0, bound=0.2):
        super().__init__(mu, sigma)
        self.bound = bound

    def __call__(self, size):
        return np.clip(np.random.normal(self.mu, self.sigma, size), -self.bound, self.bound)

    def __repr__(self):
        return 'ClippedNormalActionNoise(mu={}, sigma={}, bound={})'.format(self.mu, self.sigma, self.bound)


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu=0.0, sigma=0.2, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self, size):
        self.x_prev = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=size)
        return self.x_prev

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else 0.

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
