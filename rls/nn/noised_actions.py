#!/usr/bin/env python3
# encoding: utf-8

import math
from abc import ABC, abstractmethod

import torch as th

Noise_action_REGISTER = {}


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


class NoisedAction(ABC, object):

    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    @abstractmethod
    def __call__(self, size):
        raise NotImplementedError


class NormalNoisedAction(NoisedAction):
    def __init__(self, mu=0.0, sigma=1.0, action_bound=1.0):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.action_bound = action_bound

    def __call__(self, action):
        return (action + th.normal(self.mu, self.sigma, action.shape)).clamp(-self.action_bound, self.action_bound)

    def __repr__(self):
        return 'NormalNoisedAction(mu={}, sigma={}, action_bound={})'.format(self.mu, self.sigma, self.action_bound)


class ClippedNormalNoisedAction(NormalNoisedAction):
    def __init__(self, mu=0.0, sigma=1.0, action_bound=1.0, noise_bound=0.2):
        super().__init__(mu, sigma, action_bound)
        self.noise_bound = noise_bound

    def __call__(self, action):
        return (action + th.normal(self.mu, self.sigma, action.shape).clamp(-self.noise_bound, self.noise_bound)).clamp(
            -self.action_bound, self.action_bound)

    def __repr__(self):
        return 'ClippedNormalNoisedAction(mu={}, sigma={}, action_bound={}, noise_bound={})'.format(self.mu, self.sigma,
                                                                                                    self.action_bound,
                                                                                                    self.noise_bound)


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckNoisedAction(NormalNoisedAction):
    def __init__(self, mu=0.0, sigma=0.2, action_bound=1.0, theta=.15, dt=1e-2, x0=None):
        super().__init__(mu, sigma, action_bound)
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self, action):
        self.x_prev = self.x_prev + self.theta * \
                      (self.mu - self.x_prev) * self.dt + self.sigma * \
                      math.sqrt(self.dt) * th.randn(action.shape)
        return (action + self.x_prev).clamp(-self.action_bound, self.action_bound)

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else 0.

    def __repr__(self):
        return 'OrnsteinUhlenbeckNoisedAction(mu={}, sigma={}, action_bound={}, theta={}, dt={})'.format(self.mu,
                                                                                                         self.sigma,
                                                                                                         self.action_bound,
                                                                                                         self.theta,
                                                                                                         self.dt)


Noise_action_REGISTER['normal'] = NormalNoisedAction
Noise_action_REGISTER['clip_normal'] = ClippedNormalNoisedAction
Noise_action_REGISTER['ou'] = OrnsteinUhlenbeckNoisedAction
