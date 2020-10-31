#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from rls.utils.np_utils import (int2one_hot,
                                discounted_sum,
                                discounted_sum_minus,
                                normalization,
                                standardization)


class DataBuffer(object):
    '''
    On-policy 算法的经验池
    '''

    def __init__(self,
                 n_agents=1,
                 dict_keys=['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done'],
                 rnn_3dim_keys=['s', 's_', 'visual_s', 'visual_s_']):
        '''
        params:
            dict_keys: 要存入buffer中元素的名称
            n_agents: 一个policy控制的智能体数量
            rnn_3dim_keys: 如果使用rnn训练时，从经验池取出的数据中需要设置为[batchsize, timestep, dimension]的元素名称
        '''
        assert n_agents > 0
        self.n_agents = n_agents
        self.dict_keys = dict_keys
        self.buffer = {k: [] for k in dict_keys}
        self.rnn_3dim_keys = rnn_3dim_keys
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
        dc_r = discounted_sum(self.buffer['r'], gamma, init_value, self.buffer['done'])
        if normalize:
            dc_r = standardization(np.asarray(dc_r))
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
        self.buffer['td_error'] = discounted_sum_minus(
            self.buffer['r'],
            gamma,
            init_value,
            self.buffer['done'],
            self.buffer['value']
        )

    def cal_gae_adv(self, lambda_, gamma, normalize=False):
        '''
        计算GAE优势估计
        adv = td(s) + gamma * lambda * (1 - done) * td(s')
        '''
        assert 'td_error' in self.buffer.keys()
        adv = np.asarray(discounted_sum(
            self.buffer['td_error'],
            lambda_ * gamma,
            0,
            self.buffer['done']
        ))
        if normalize:
            adv = standardization(adv)
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
        '''
        create sampling data iterator without using rnn.

        params:
            batch_size: the batch size of training data
            keys: the keys of data that should be sampled to train policies
        return:
            sampled data.
        '''
        keys = keys or self.buffer.keys()
        keys_shape = self.calculate_dim_before_sample(keys)
        all_data = [np.vstack(self.buffer[k]).reshape(self.eps_len * self.n_agents, *keys_shape[k]).astype(np.float32) for k in keys]
        idxs = np.arange(self.eps_len * self.n_agents)
        np.random.shuffle(idxs)
        for i in range(0, self.eps_len * self.n_agents, batch_size * self.n_agents):
            _idxs = idxs[i:i + batch_size * self.n_agents]
            yield [data[_idxs] for data in all_data]

    def sample_generater_rnn(self, time_step, keys=None):
        '''
        create rnn sampling data iterator.

        params:
            time_step: the length of time slide window
            keys: the keys of data that should be sampled to train policies
        return:
            sampled data.
            if key in self.rnn_3dim_keys, then return data with shape (agents_num, time_step, *)
            else return with shape (agents_num*time_step, *)
        '''
        keys = keys or self.buffer.keys()
        # acquire shape of each items
        keys_shape = self.calculate_dim_before_sample(keys)
        all_data = [np.vstack(self.buffer[k]).reshape(self.eps_len, self.n_agents, -1).astype(np.float32) for k in keys]
        # change data shape from [timestep, agents, dim] to [agents, timestep, dim]
        # TODO: check whether suit for visual observations.
        all_data = [np.transpose(data, (1, 0, 2)) for data in all_data]
        # reshape necessary data shape from [agents, timestep, dim] to [agents*timestep, dim], i.e. rewards
        all_data = [np.reshape(data, (self.n_agents, self.eps_len, *keys_shape[k])) if k in self.rnn_3dim_keys
                    else np.reshape(data, (self.n_agents * self.eps_len, *keys_shape[k]))
                    for k, data in zip(keys, all_data)]
        idxs = np.arange(0, self.eps_len, time_step)    # [1, 3, 5, 7, 9]
        np.random.shuffle(idxs)
        for i in idxs:
            yield [data[:, i:i + time_step] if k in self.rnn_3dim_keys else data[i * self.n_agents:(i + time_step) * self.n_agents]
                   for k, data in zip(keys, all_data)]

    def get_curiosity_data(self):
        '''
        返回用于好奇心机制的数据
        '''
        keys = ['s', 'visual_s', 'a', 'r', 's_', 'visual_s_']
        for k in keys:
            if k not in self.buffer.keys():
                raise Exception('Buffer does not has key {k}.')
        keys_shape = self.calculate_dim_before_sample(keys)
        all_data = [np.vstack(self.buffer[k]).reshape(self.eps_len * self.n_agents, *keys_shape[k]).astype(np.float32) for k in keys]
        return all_data

    def convert_action2one_hot(self, a_counts):
        '''
        用于在训练前将buffer中的离散动作的索引转换为one_hot类型
        '''
        if 'a' in self.buffer.keys():
            self.buffer['a'] = [int2one_hot(a.astype(np.int32), a_counts) for a in self.buffer['a']]

    def clear(self):
        '''
        清空临时存储经验池
        '''
        self.eps_len = 0
        for k in self.buffer.keys():
            self.buffer[k].clear()

    def __getattr__(self, name):
        '''
        TODO: Annotation
        '''
        return self.buffer[name]

    def __getitem__(self, name):
        '''
        TODO: Annotation
        '''
        return self.buffer[name]

    def normalize_vector_obs(self, func):
        '''
        TODO: Annotation
        '''
        if 's' in self.buffer.keys():
            self.buffer['s'] = [func(s) for s in self.buffer['s']]
        if 's_' in self.buffer.keys():
            self.buffer['s_'] = [func(s) for s in self.buffer['s_']]

    def calculate_dim_before_sample(self, keys=None):
        '''
        calculate the dimension of each items stored in data buffer. This will help to reshape the data.
        For example, if the dimension of vector obs is 4, and the dimension of visual obs is (84, 84, 3),
        you cannot just use tf.reshape(x, (batch_size, time_step, -1)) to get the correct shape of visual obs.
        By recording the data dimension in this way, it is easy to reshape them.

        params: 
            keys: the key of items in data buffer
        return:
            keys_shape: a dict that include all dimension info of each key in data buffer
        '''
        keys = keys or self.buffer.keys()
        keys_shape = {k: self.buffer[k][0].shape[1:] if len(self.buffer[k][0].shape[1:]) > 0 else (-1,) for k in keys}
        return keys_shape

    def __str__(self):
        return str(self.buffer)


if __name__ == "__main__":
    db = DataBuffer(
        dict_keys=['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done'],
        n_agents=2,
        rnn_3dim_keys=['s', 's_', 'visual_s', 'visual_s_'])

    for i in range(10):
        db.add(
            np.full((2, 2), i, dtype=np.float32),
            np.full((2, 8, 8, 3), i, dtype=np.float32),
            np.full((2, 2), i, dtype=np.float32),
            np.full((2,), i, dtype=np.float32),
            np.full((2, 2), i, dtype=np.float32),
            np.full((2, 8, 8, 3), i, dtype=np.float32),
            np.full((2,), False, dtype=np.float32)
        )

    # should be [[9, 9], [9, 9]]
    print(db.last_s())
    # shouble be np.full((2, 8, 8, 3), 9)
    print(db.last_visual_s())

    for d in db.sample_generater(batch_size=2, keys=['s', 'r']):
        print(d[0].shape, d[1].shape)

    for d in db.sample_generater_rnn(time_step=6, keys=['s', 'r']):
        print(d[0].shape, d[1].shape)
        # print(d[0])

    # db.cal_dc_r(1., 1.)
    # print(db.buffer['discounted_reward'])

    # db.cal_dc_r(1., 1., normalize=True)
    # print(db.buffer['discounted_reward'])

    # db.clear()
    # print(db)
