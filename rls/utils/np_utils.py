#!/usr/bin/env python3
# encoding: utf-8

import itertools
import scipy.signal
import numpy as np


def intprod(x):
    return int(np.prod(x))


def discounted_sum(x, gamma, init_value, dones):
    l = len(x)
    out = [0] * l
    for i in reversed(range(l)):
        out[i] = init_value = x[i] + init_value * gamma * (1 - dones[i])
    return out


def calculate_td_error(r, gamma, d, v, v_):
    r = np.array(r)
    v = np.array(v)
    d = np.array(d)
    v_ = np.array(v_)
    td = r + gamma * (1 - d) * v_ - v
    return td


def get_discrete_action_list(action_dim_list):
    """
    input: [3, 2, 2]
    output: 
        [[0 0 0]
        [0 0 1]
        [0 1 0]
        [0 1 1]
        [1 0 0]
        [1 0 1]
        [1 1 0]
        [1 1 1]
        [2 0 0]
        [2 0 1]
        [2 1 0]
        [2 1 1]]
    """
    return np.array(list(itertools.product(*[list(range(l)) for l in action_dim_list])))


def int2one_hot(x, act_nums):
    '''
    params:
        x: 1D or 2D
    input: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 12
    output: [[1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
            [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]]
    '''
    x = np.ravel(x)
    a = np.eye(act_nums)[x]
    return a


def all_equal(x):
    '''
    return whether items in x all equal or not.
    '''
    return (x == x.reshape(-1)[0]).all()


def get_first_item(x):
    '''
    return the first item in numpy array.
    '''
    return x.reshape(-1)[0]


def is_inf_inside(x):
    '''
    return whether np.inf, -np.inf is in x or not.
    '''
    return np.isinf(x).any()


def arrprint(x, n):
    assert isinstance(x, np.ndarray)
    return ', '.join([str(f'%{4+n}.{n}f' % i) for i in sorted(x)])


def normalization(data):
    '''
    归一化，规范化，规范化给定数据集中的所有数值(或者分别对每个feature列处理)属性值，类属性除外。结果值默认在区间[0,1]，但是利用缩放和平移参数，我们能将数值属性值规范到任何区间。
    data -> [0, 1]
    '''
    assert isinstance(data, np.ndarray)
    _min = np.min(data)
    _max = np.max(data)
    return (data - _min) / (_max - _min)


def normalization_neg(data):
    '''
    归一化，规范化，规范化给定数据集中的所有数值(或者分别对每个feature列处理)属性值，类属性除外。结果值默认在区间[0,1]，但是利用缩放和平移参数，我们能将数值属性值规范到任何区间。
    data -> [-1, 1]
    '''
    assert isinstance(data, np.ndarray)
    _max = np.max(abs(data))
    return data / _max


def standardization(data):
    '''
    标准化，标准化给定数据集中所有数值属性(或者分别对每个feature列处理)的值到一个0均值和单位方差的正态分布。
    data -> Normal(0, 1)
    '''
    assert isinstance(data, np.ndarray)
    mu = np.mean(data)
    sigma = np.std(data) + 1e-8
    return (data - mu) / sigma


class SMA:
    '''
    Simple Moving Average
    '''

    def __init__(self, n):
        self.n = n
        self.now = 0
        self.r_list = []
        self.max, self.min, self.mean = 0, 0, 0

    def update(self, r):
        assert isinstance(r, (np.ndarray, list)), 'r must have __len__ attr'
        r = np.array(r)
        self.r_list.append(r)
        if self.now >= self.n:
            r_old = self.r_list.pop(0)
            self.max += (r.max() - r_old.max()) / self.n
            self.min += (r.min() - r_old.min()) / self.n
            self.mean += (r.mean() - r_old.mean()) / self.n
        else:
            self.now = min(self.now + 1, self.n)
            self.max += (r.max() - self.max) / self.now
            self.min += (r.min() - self.min) / self.now
            self.mean += (r.mean() - self.mean) / self.now

    @property
    def rs(self):
        return dict([
            ['sma_max', self.max],
            ['sma_min', self.min],
            ['sma_mean', self.mean]
        ])


if __name__ == "__main__":
    a = SMA(10)
    for i in range(20):
        a.update([i, i + 1, i + 2])
        print(i, a.r_list)
        print(a.now)
        print(a.rs)
