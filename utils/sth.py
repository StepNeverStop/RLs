import numpy as np

class sth(object):
    @staticmethod
    def discounted_sum(x, gamma, init_value, dones):
        assert isinstance(x,np.ndarray)
        assert isinstance(dones,np.ndarray)

        y = []
        for i in reversed(range(len(x))):
            init_value = gamma * (1 - dones[i]) * init_value + x[i]
            y.insert(0, init_value)
        return y

    @staticmethod
    def discounted_sum_minus(x, gamma, init_value, dones, z):
        assert isinstance(x,np.ndarray)
        assert isinstance(dones,np.ndarray)
        assert isinstance(z,np.ndarray)

        y = []
        for i in reversed(range(len(x))):
            y.insert(0, gamma * (1 - dones[i]) * init_value + x[i] - z[i])
            init_value = z[i]
        return y
