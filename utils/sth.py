import os
import itertools
import numpy as np


class sth(object):

    @staticmethod
    def discounted_sum(x, gamma, init_value, dones):
        y = []
        for _x, _d in zip(x[::-1], dones[::-1]):
            init_value = gamma * (1 - _d) * init_value + _x
            y.append(init_value)
        return y[::-1]

    @staticmethod
    def discounted_sum_minus(x, gamma, init_value, dones, z):
        y = []
        for _x, _d, _z in zip(x[::-1], dones[::-1], z[::-1]):
            y.append(gamma * (1 - _d) * init_value + _x - _z)
            init_value = _z
        return y[::-1]

    @staticmethod
    def int2action_index(x, action_dim_list):
        """
        input: [0,1,2,3,4,5,6,7,8,9,10,11], [3, 2, 2]
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
        x = np.array(x)
        index_list = list(itertools.product(*[list(range(l)) for l in action_dim_list]))
        return np.asarray(index_list)[x]

    @staticmethod
    def int2one_hot(x, action_dim_prod):
        '''
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
        x = np.asarray(x).flatten()
        a = np.eye(action_dim_prod)[x]
        return a
