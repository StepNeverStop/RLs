import numpy as np
from utils.sth import sth
from utils.np_utils import normalization, standardization

class DataBuffer(object):
    '''
    On-policy 算法的经验池
    '''

    def __init__(self, dict_keys=['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done']):
        self.dict_keys = dict_keys
        self.buffer = dict([
            [n, []] for n in dict_keys
        ])
        self.eps_len = 0

    def add(self, *args):
        '''
        添加数据
        '''
        [self.buffer[k].append(arg) for k, arg in zip(self.dict_keys, args)]
        self.eps_len += 1

    def cal_dc_r(self, gamma, init_value, normalize=False):
        '''
        计算折扣奖励
        param gamma: 折扣因子 gamma \in [0, 1)
        param init_value: 序列最后状态的值
        '''
        dc_r = sth.discounted_sum(self.buffer['r'], gamma, init_value, self.buffer['done'])
        if normalize:
            dc_r -= np.mean(dc_r)
            dc_r /= np.std(dc_r)
        self.buffer['discounted_reward'] = list(dc_r)

    def cal_tr(self, init_value):
        '''
        计算总奖励
        '''
        self.buffer['total_reward'] = self.cal_dc_r(1., init_value)

    def cal_td_error(self, gamma, init_value):
        '''
        计算td error
        TD = r + gamma * (1- done) * v(s') - v(s)
        '''
        assert 'value' in self.buffer.keys()
        self.buffer['td_error'] = sth.discounted_sum_minus(
            self.buffer['r'],
            gamma,
            init_value,
            self.buffer['done'],
            self.buffer['value']
        )

    def cal_gae_adv(self, lambda_, gamma):
        '''
        计算GAE优势估计
        adv = td(s) + gamma * lambda * (1 - done) * td(s')
        '''
        assert 'td_error' in self.buffer.keys()
        adv = np.asarray(sth.discounted_sum(
            self.buffer['td_error'],
            lambda_ * gamma,
            0,
            self.buffer['done']
        ))
        self.buffer['gae_adv'] = list(standardization(adv))

    def last_s(self):
        '''
        获取序列末尾的状态，即s_[-1]
        '''
        assert 's_' in self.buffer.keys()
        return self.buffer['s_'][-1]

    def last_visual_s(self):
        '''
        获取序列末尾的图像，即visual_s_[-1]
        '''
        assert 'visual_s_' in self.buffer.keys()
        return self.buffer['visual_s_'][-1]

    def sample_generater(self, batch_size, keys=None):
        if keys is None:
            keys = self.buffer.keys()
        agents_num = len(self.buffer['s'][0])
        all_data = [np.vstack(self.buffer[k]).reshape(self.eps_len*agents_num, -1).astype(np.float32) for k in keys]
        for i in range(0, self.eps_len*agents_num, batch_size):
            yield [data[i:i+batch_size] for data in all_data]

    def get_curiosity_data(self):
        '''
        返回用于好奇心机制的数据
        '''
        keys = ['s', 'visual_s', 'a', 'r', 's_', 'visual_s_']
        for k in keys:
            if k not in self.buffer.keys():
                raise Exception('Buffer does not has key {k}.')
        agents_num = len(self.buffer['s'][0])
        all_data = [np.vstack(self.buffer[k]).reshape(self.eps_len*agents_num, -1).astype(np.float32) for k in keys]
        return all_data

    def convert_action2one_hot(self, a_counts):
        '''
        用于在训练前将buffer中的离散动作的索引转换为one_hot类型
        '''
        if 'a' in self.buffer.keys():
            self.buffer['a'] = [sth.int2one_hot(a.astype(np.int32), a_counts) for a in self.buffer['a']]

        
    def clear(self):
        '''
        清空临时存储经验池
        '''
        self.eps_len = 0
        for k in self.buffer.keys():
            self.buffer[k].clear()

    def __getattr__(self, name):
        return self.buffer[name]

    def __getitem__(self, name):
        return self.buffer[name]