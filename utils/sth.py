import os
import yaml
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
    
    @staticmethod
    def save_config(dir_name, config):
        fw = open(os.path.join(dir_name, 'config.yaml'), 'w', encoding='utf-8')
        yaml.dump(config, fw)
        fw.close()
        print(f'save config to {dir_name}')

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
