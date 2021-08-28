#!/usr/bin/env python3
# encoding: utf-8

import math
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import torch as t


def clip_nn_log_std(log_std, _min=-20, _max=2):
    """
    scale log_std from [-1, 1] to [_min, _max]
    Args:
        log_std: log standard deviation of a gaussian distribution
        _min: corrected minimum
        _max: corrected maximum
    Return:
        log_std after scaling, range from _min to _max
    """
    return _min + 0.5 * (_max - _min) * (log_std + 1)


def gaussian_likelihood(x, mu, log_std):
    """
    Calculating the log probability of a sample from gaussian distribution.
    Args:
        x: sample data from Normal(mu, std)
        mu: mean of the gaussian distribution
        log_std: log standard deviation of the gaussian distribution
    Return:
        log probability of sample. i.e. [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]], not [[0.3], [0.3]]
    """
    pre_sum = -0.5 * (((x - mu) / (log_std.exp() + t.finfo().eps))
                      ** 2 + 2 * log_std + math.log(2 * np.pi))
    return t.maximum(pre_sum, t.full_like(pre_sum, t.finfo().eps))


def gaussian_entropy(log_std):
    '''
    Calculating the entropy of a Gaussian distribution.
    Args:
        log_std: log standard deviation of the gaussian distribution.
    Return:
        The average entropy of a batch of data.
    '''
    return (0.5 * (1 + (2 * np.pi * log_std.exp()**2 + t.finfo().eps).log())).mean()


def squash_action(pi, log_pi, *, is_independent=True):
    """
    Enforcing action bounds.
    squash action to range [-1, 1] and calculate the correct log probability value.
    Args:
        pi: sample of gaussian distribution
        log_pi: log probability of the sample
    Return:
        sample range of [-1, 1] after squashed.
        the corrected log probability of squashed sample.
    """
    pi.tanh_()
    sub = (clip_but_pass_gradient(1 - pi**2, l=0, h=1) + t.finfo().eps).log()
    log_pi -= sub
    if is_independent:
        log_pi = log_pi.sum(-1, keepdim=True)
    return pi, log_pi


def clip_but_pass_gradient(x, l=-1., h=1.):
    """
    Stole this function from SpinningUp
    Args:
        x: data to be clipped.
        l: lower bound
        h: upper bound.
    Return:
        if x < l:
            l
        elif x > h:
            h
        else:
            x
    """
    clip_up = t.as_tensor(x > h)
    clip_low = t.as_tensor(x < l)
    return x + ((h - x) * clip_up + (l - x) * clip_low).detach()


def tsallis_entropy_log_q(log_pi, q):
    if q == 1.:  # same to SAC
        return log_pi.sum(-1, keepdim=True)
    else:
        if q > 0.:
            '''
            cite from original implementation: https://github.com/rllab-snu/tsallis_actor_critic_mujoco/blob/9f9ba8e4dc8f9680f1e516d3b1391c9ded3934e3/spinup/algos/tac/core.py#L47
            '''
            pi_p = log_pi.exp()
        else:
            pi_p = t.minimum(log_pi.exp(), t.pow(10., 8 / (1 - q)))
        safe_x = pi_p.maximum(t.full_like(pi_p, t.finfo().eps))
        log_q_pi = (safe_x.pow(1 - q) - 1) / (1 - q)
        return log_q_pi.sum(-1, keepdim=True)


def sync_params(tge: t.nn.Module, src: t.nn.Module, ployak: float = 0.) -> NoReturn:
    '''
    update weights of target neural network.
    ployak = 1 - tau
    '''
    for _t, _s in zip(tge.parameters(), src.parameters()):
        _t.data.copy_(_t.data * ployak + _s.data * (1. - ployak))


def sync_params_list(nets_list: List[Union[List, Tuple]], ployak: float = 0.) -> NoReturn:
    '''
    update weights of target neural network.
    ployak = 1 - tau
    '''
    for tge, src in zip(*nets_list):
        sync_params(tge, src, ployak)


def q_target_func(reward, gamma, done, q_next, begin_mask,
                  nstep=None, detach=True):
    '''
    params:
        reward: [T, B, 1],
        gamma: float
        done: [T, B, 1]
        q_next: [T, B, 1]
        begin_mask: [T, B, 1]
    return:
        q_value: [T, B, 1]
    '''
    # print(reward.shape, done.shape, q_next.shape, begin_mask.shape)
    n_step = reward.shape[0]
    # TODO: optimize
    if nstep is not None:
        q_target = t.zeros_like(q_next)  # [T, B, 1]
        q_post = q_next[-1]
        for _t in range(n_step)[::-1]:
            q_target[_t] = reward[_t] + gamma * (1 - done[_t]) * q_post
            q_post = t.where(begin_mask[_t] > 0,
                             q_next[max(_t-1, 0)], q_target[_t])
    elif nstep == 1:
        q_target = t.zeros_like(q_next)  # [T, B, 1]
        for _t in range(n_step):
            q_target[_t] = reward[_t] + gamma * (1 - done[_t]) * q_next[_t]
    else:
        raise NotImplementedError
    return q_target.detach() if detach else q_target
