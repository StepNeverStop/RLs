import os
import numpy as np


class sth(object):
    @staticmethod
    def discounted_sum(x, gamma, init_value, dones):
        assert isinstance(x, np.ndarray), 'type of sth.discounted_sum.x must be np.ndarray'
        assert isinstance(dones, np.ndarray), 'type of sth.discounted_sum.done must be np.ndarray'

        y = []
        for i in reversed(range(len(x))):
            init_value = gamma * (1 - dones[i]) * init_value + x[i]
            y.insert(0, init_value)
        return y

    @staticmethod
    def discounted_sum_minus(x, gamma, init_value, dones, z):
        assert isinstance(x, np.ndarray), 'type of sth.discounted_sum_minus.x must be np.ndarray'
        assert isinstance(dones, np.ndarray), 'type of sth.discounted_sum_minus.dones must be np.ndarray'
        assert isinstance(z, np.ndarray), 'type of sth.discounted_sum_minus.z must be np.ndarray'

        y = []
        for i in reversed(range(len(x))):
            y.insert(0, gamma * (1 - dones[i]) * init_value + x[i] - z[i])
            init_value = z[i]
        return y


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
        y = []
        x = np.squeeze(x)
        for i in reversed(action_dim_list):
            y.insert(0, x % i)
            x //= i
        return np.array(y).T

    @staticmethod
    def action_index2int(z, action_dim_list):
        '''
        input: [[0 0 0]
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
                [2 1 1]], [3, 2, 2]
        output: [ 0  1  2  3  4  5  6  7  8  9 10 11]
        '''
        assert isinstance(z, np.ndarray), 'type of sth.action_index2int.z must be np.ndarray'
        if len(z.shape) == 1:
            z = z[np.newaxis, :]
        x = []
        y = 1
        for i in list(reversed(action_dim_list)):
            x.insert(0, y)
            y *= i
        return z.dot(np.array(x))

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
        if hasattr(x, '__len__'):
            a = np.zeros([len(x), action_dim_prod])
            for i in range(len(x)):
                a[i, x[i]] = 1
        else:
            a = np.zeros(action_dim_prod)
            a[x] = 1
        return a

    @staticmethod
    def action_index2one_hot(index, action_dim_list):
        '''
        input: [[0 0 0]
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
                [2 1 1]], [3, 2, 2]
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
        assert isinstance(index, np.ndarray), 'type of sth.action_index2one_hot.index must be np.ndarray'
        if len(index.shape) == 1:
            index = index[:, np.newaxis]
        return sth.int2one_hot(sth.action_index2int(index, action_dim_list), np.asarray(action_dim_list).prod())

    @staticmethod
    def get_batch_one_hot(action, action_multiplication_factor, cols):
        """
        input: [[2, 1],[2, 0]], [3, 1], 9
        output: [[0, 0, 0, 0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0, 0]]
        """
        assert isinstance(action, np.ndarray), 'type of sth.get_batch_one_hot.action must be np.ndarray'
        assert isinstance(action_multiplication_factor, np.ndarray), 'type of sth.get_batch_one_hot.action_multiplication_factor must be np.ndarray'
        ints = action.dot(action_multiplication_factor)
        x = np.zeros([action.shape[0], cols])
        for i, j in enumerate(ints):
            x[i, j] = 1
        return x

    @staticmethod
    def action_index2action_value(action_index, action_dim_list):
        """
        let actions' value between -1 and 1, if action_lict is [3,3], means that every dimension has 3 actions average from -1 to 1, like [-1, 0, 1], so index [0, 2] means action value [-1, 1]
        input: [0, 2], [3, 3]
        output: [-1, 1]
        """
        assert isinstance(action, np.ndarray), 'type of sth.action_index2action_value.action must be np.ndarray'
        assert 1 not in action_dim_list, 'sth.action_index2action_value.action_dim_list must not include 1'
        return 2 / (np.array([action_dim_list]) - 1) * action_index - 1
