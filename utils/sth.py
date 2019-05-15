import os
import yaml
import numpy as np


class sth(object):
    @staticmethod
    def discounted_sum(x, gamma, init_value, dones):
        assert isinstance(x, np.ndarray)
        assert isinstance(dones, np.ndarray)

        y = []
        for i in reversed(range(len(x))):
            init_value = gamma * (1 - dones[i]) * init_value + x[i]
            y.insert(0, init_value)
        return y

    @staticmethod
    def discounted_sum_minus(x, gamma, init_value, dones, z):
        assert isinstance(x, np.ndarray)
        assert isinstance(dones, np.ndarray)
        assert isinstance(z, np.ndarray)

        y = []
        for i in reversed(range(len(x))):
            y.insert(0, gamma * (1 - dones[i]) * init_value + x[i] - z[i])
            init_value = z[i]
        return y

    @staticmethod
    def save_config(dicpath, config):
        if not os.path.exists(dicpath):
            os.makedirs(dicpath)
        fw = open(os.path.join(dicpath, 'config.yaml'), 'w', encoding='utf-8')
        yaml.dump(config, fw)
        fw.close()
        print(f'save config to {dicpath}')

    @staticmethod
    def load_config(filename):
        if os.path.exists(filename):
            f = open(filename, 'r', encoding='utf-8')
        else:
            raise Exception('cannot find this config.')
        x = yaml.safe_load(f.read())
        f.close()
        print(f'load config from {filename}')
        return x

    @staticmethod
    def get_action_multiplication_factor(action_list):
        """
        input: [3, 2, 2]
        output: [4, 2, 1]
        """
        x = []
        y = 1
        for i in list(reversed(action_list)):
            x.insert(0, y)
            y *= i
        return np.array(x)

    @staticmethod
    def int2action_index(x, action_multiplication_factor):
        """
        input: [7], [3,1]
        output: [2,1]
        """
        assert isinstance(x, np.ndarray)
        y = []
        for i in action_multiplication_factor[:-1]:
            y.append(x // i)
            x %= i
        y.append(x)
        return np.stack(y, axis=1)

    @staticmethod
    def get_batch_one_hot(action, action_multiplication_factor, cols):
        """
        input: [2, 1], [3, 1], 9
        output: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        """
        assert isinstance(action, np.ndarray)
        assert isinstance(action_multiplication_factor, np.ndarray)
        ints = action.dot(action_multiplication_factor)
        x = np.zeros([action.shape[0], cols])
        for i, j in enumerate(ints):
            x[i, j] = 1
        return x

    @staticmethod
    def index2action(action_index, action_list):
        """
        input: [0, 2], [3, 3]
        output: [-1, 1]
        """
        assert isinstance(action, np.ndarray)
        assert 1 not in action_list
        return 2 / (np.array([action_list]) - 1) * action_index - 1
