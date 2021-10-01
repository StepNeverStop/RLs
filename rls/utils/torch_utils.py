#!/usr/bin/env python3
# encoding: utf-8

import math
from typing import List, NoReturn, Tuple, Union

import numpy as np
import torch as th
import torch.nn as nn


def grads_flatten(loss, model, **kwargs):
    grads = th.autograd.grad(loss, model.parameters(), **kwargs)
    return th.cat([grad.reshape(-1) for grad in grads])


def set_from_flat_params(model, flat_params):
    prev_ind = 0
    for name, param in model.named_parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size
    return model


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
    pre_sum = -0.5 * (((x - mu) / (log_std.exp() + th.finfo().eps))
                      ** 2 + 2 * log_std + math.log(2 * np.pi))
    return th.maximum(pre_sum, th.full_like(pre_sum, th.finfo().eps))


def gaussian_entropy(log_std):
    """
    Calculating the entropy of a Gaussian distribution.
    Args:
        log_std: log standard deviation of the gaussian distribution.
    Return:
        The average entropy of a batch of data.
    """
    return (0.5 * (1 + (2 * np.pi * log_std.exp() ** 2 + th.finfo().eps).log())).mean()


def squash_action(pi, log_pi, *, is_independent=True):
    """
    Enforcing action bounds.
    squash action to range [-1, 1] and calculate the correct log probability value.
    Args:
        pi: sample of gaussian distribution
        log_pi: log probability of the sample
        is_independent: todo
    Return:
        sample range of [-1, 1] after squashed.
        the corrected log probability of squashed sample.
    """
    pi.tanh_()
    sub = (clip_but_pass_gradient(1 - pi ** 2, l=0, h=1) + th.finfo().eps).log()
    log_pi = log_pi - sub
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
    clip_up = th.as_tensor(x > h)
    clip_low = th.as_tensor(x < l)
    return x + ((h - x) * clip_up + (l - x) * clip_low).detach()


def tsallis_entropy_log_q(log_pi, q):
    if q == 1.:  # same to SAC
        return log_pi.sum(-1, keepdim=True)
    else:
        if q > 0.:
            """
            cite from original implementation: https://github.com/rllab-snu/tsallis_actor_critic_mujoco/blob/9f9ba8e4dc8f9680f1e516d3b1391c9ded3934e3/spinup/algos/tac/core.py#L47
            """
            pi_p = log_pi.exp()
        else:
            pi_p = th.minimum(log_pi.exp(), th.pow(10., 8 / (1 - q)))
        safe_x = pi_p.maximum(th.full_like(pi_p, th.finfo().eps))
        log_q_pi = (safe_x.pow(1 - q) - 1) / (1 - q)
        return log_q_pi.sum(-1, keepdim=True)


def sync_params(tge: nn.Module, src: nn.Module, polyak: float = 0.) -> NoReturn:
    """
    update weights of target neural network.
    polyak = 1 - tau
    """
    for _t, _s in zip(tge.parameters(), src.parameters()):
        _t.data.copy_(_t.data * polyak + _s.data * (1. - polyak))


def sync_params_list(nets_list: List[Union[List, Tuple]], polyak: float = 0.) -> NoReturn:
    """
    update weights of target neural network.
    polyak = 1 - tau
    """
    for tge, src in zip(*nets_list):
        sync_params(tge, src, polyak)


def q_target_func(reward, gamma, done, q_next, begin_mask,
                  nstep=None, detach=True):
    """ TODO: under remove
    params:
        reward: [T, B, 1],
        gamma: float
        done: [T, B, 1]
        q_next: [T, B, 1]
        begin_mask: [T, B, 1]
    return:
        q_value: [T, B, 1]
    """
    # print(reward.shape, done.shape, q_next.shape, begin_mask.shape)
    n_step = reward.shape[0]
    # TODO: optimize
    if nstep is None:
        q_target = th.zeros_like(q_next)  # [T, B, 1]
        q_post = q_next[-1]
        for t in range(n_step)[::-1]:
            q_target[t] = reward[t] + gamma * (1 - done[t]) * q_post
            q_post = th.where(begin_mask[t] > 0,
                              q_next[max(t - 1, 0)], q_target[t])
    elif nstep == 1:
        q_target = th.zeros_like(q_next)  # [T, B, 1]
        for t in range(n_step):
            q_target[t] = reward[t] + gamma * (1 - done[t]) * q_next[t]
    else:
        raise NotImplementedError
    return q_target.detach() if detach else q_target


def n_step_return(reward, gamma, done, q_next, begin_mask=None,
                  nstep=None, terminal_idxs=None,
                  ret_all=False):
    """
    params:
        reward: [T, B, 1],
        gamma: float
        done: [T, B, 1]
        q_next: [T, B, 1]
        begin_mask: [T, B, 1]
    return:
        q_value: [T, B, 1]
    """
    T = reward.shape[0]
    if nstep is None:
        nstep = T
    else:
        nstep = min(nstep, T)

    q_values = q_next.clone()

    if begin_mask is None:
        begin_mask = th.zeros_like(reward)

    if terminal_idxs is not None:
        if terminal_idxs == 0 or terminal_idxs + nstep <= T:  # == is ok
            return []
        else:
            for i in range(terminal_idxs):
                q_values[i] = reward[i] + gamma * (1 - done[i]) * q_next[i + 1] * (1 - begin_mask[i + 1])
    else:
        terminal_idxs = T
        for i in range(terminal_idxs):
            q_values[i] = reward[i] + gamma * (1 - done[i]) * q_next[i]

    rets = [q_values]
    rets.extend(
        n_step_return(reward, gamma, done, q_values, begin_mask, nstep, terminal_idxs - 1, ret_all=True)
    )
    if ret_all:
        return rets
    else:
        return rets[-1]


def td_lambda_return(reward, gamma, done, q_next, begin_mask, _lambda=0.9):
    """
    _lambda \in [0, 1], 0 for TD(0), 1 for MC
    **Strong Recommend long time steps with large _lambda**, 'cause short time steps
    may cause incorrect calculation of TD(\lambda) due to the non-done state.
    """
    n_step_returns = n_step_return(
        reward, gamma, done, q_next, begin_mask, ret_all=True)
    L = len(n_step_returns)

    """
    record which experience will encounter done flag when calculating n-step return.
    'cause there will be different formula for calculating the last term of TD(\lambda) 
    with different situation of time step t+n (done or not).
    """
    roll_done = done.clone()  # [T, B, 1]
    for i in range(L):
        done += roll_done
        roll_done = th.roll(roll_done, -1, 0)
        roll_done[-1] = 0.

    q_values = th.zeros_like(q_next)
    for i in range(L - 1):  # [1step, ..., nstep]
        q_values += (1 - _lambda) * (_lambda ** i) * n_step_returns[i]
    """
    For experience that when calculating n-step return encountered done flag, we should multiply 
    \lambda^{n-1} to the last term G^{n}_{T}. But when not encountered done flag, we should multiply
    (1-\lambda)*\lambda^{n-1} to the last term G^{n}_{t+n}.
    """
    q_values += th.where(done > 0., 1., (1 - _lambda)) * \
                (_lambda ** (L - 1)) * n_step_returns[-1]
    """
    Normalize the coefficient of lambda return.
    i.e. for 4-step, but not done, then
        Q(\lambda) = (1-\lambda)*Q_1 + (1-\lambda)*\lambda^{1}*Q_2 + (1-\lambda)*\lambda^{2}*Q_3 \
            + (1-\lambda)*\lambda^{3}*Q_4
        but, (1-\lambda) + (1-\lambda)*\lambda^{1} + (1-\lambda)*\lambda^{2} + (1-\lambda)*\lambda^{3} \
            not equals to 1, but equals to 1 - \lambda^{4}, so we need to normalize the coefficient by 
        deviding 1 - \lambda^{4}.
        for 4-step, but done, we don't have this problem, 'cause the value of summation is equals to 1.

    we can easily use the following function to varify this situation:
        def f(l=0.9, n=10):
            return sum([(1-l)*l**i for i in range(n)])
        f(l=0.9, n=10) = (1 - 0.9**10)
    """
    q_values /= th.where(done > 0., 1., (1 - _lambda ** L))
    return q_values
