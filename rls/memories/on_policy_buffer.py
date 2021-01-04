#!/usr/bin/env python3
# encoding: utf-8

import random
import numpy as np

from collections import defaultdict

from rls.utils.np_utils import (int2one_hot,
                                discounted_sum,
                                calculate_td_error,
                                normalization,
                                standardization)
from rls.utils.specs import (BatchExperiences,
                             NamedTupleStaticClass)


class DataBuffer(object):
    '''
    On-policy 算法的经验池
    '''

    def __init__(self,
                 n_agents: int = 1,
                 rnn_cell_nums: int = 0,
                 batch_size: int = 32,
                 rnn_time_step: int = 8,
                 store_data_type: BatchExperiences = BatchExperiences,
                 sample_data_type: BatchExperiences = BatchExperiences):
        '''
        params:
            n_agents: 一个policy控制的智能体数量
        '''
        assert n_agents > 0, "assert n_agents > 0"

        self.data_buffer = defaultdict(list)
        self.n_agents = n_agents
        self.rnn_cell_nums = rnn_cell_nums
        self.cell_state_buffer = [[] for _ in range(self.rnn_cell_nums)]
        self.eps_len = 0

        self.batch_size = batch_size
        self.rnn_time_step = rnn_time_step

        self.store_data_type = store_data_type
        self.sample_data_type = sample_data_type

    def add(self, exps: BatchExperiences):
        '''
        添加数据
        '''
        for k, v in exps._asdict().items():
            self.data_buffer[k].append(v)
        self.eps_len += 1

    def add_cell_state(self, cell_states):
        '''存储LSTM隐状态'''
        for l, cs in zip(self.cell_state_buffer, cell_states):
            l.append(cs)

    def cal_dc_r(self, gamma, init_value, normalize=False):
        '''
        计算折扣奖励
        param gamma: 折扣因子 gamma \in [0, 1)
        param init_value: 序列最后状态的值
        '''
        discounted_reward = discounted_sum(self.data_buffer['reward'],
                                           gamma,
                                           init_value,
                                           self.data_buffer['done'])
        if normalize:
            discounted_reward = standardization(np.asarray(discounted_reward))
        self.data_buffer['discounted_reward'] = list(discounted_reward)

    def cal_tr(self, init_value):
        '''
        计算总奖励
        '''
        self.data_buffer['total_reward'] = self.cal_dc_r(1., init_value)

    def cal_td_error(self, gamma, init_value):
        '''
        计算td error
        TD = r + gamma * (1- done) * v(s') - v(s)
        '''
        assert 'value' in self.data_buffer.keys(), "assert 'value' in self.data_buffer.keys()"
        self.data_buffer['td_error'] = list(calculate_td_error(
            self.data_buffer['reward'],
            gamma,
            self.data_buffer['done'],
            self.data_buffer['value'],
            self.data_buffer['value'][1:] + [init_value],
        ))

    def cal_gae_adv(self, lambda_, gamma, normalize=False):
        '''
        计算GAE优势估计
        adv = td(s) + gamma * lambda * (1 - done) * td(s')
        '''
        assert 'td_error' in self.data_buffer.keys(), "assert 'td_error' in self.data_buffer.keys()"
        adv = np.asarray(discounted_sum(
            self.data_buffer['td_error'],
            lambda_ * gamma,
            0,
            self.data_buffer['done']
        ))
        if normalize:
            adv = standardization(adv)
        self.data_buffer['gae_adv'] = list(standardization(adv))

    def last_data(self, key):
        '''
        获取序列末尾的数据
        '''
        assert key in self.data_buffer.keys(), f"assert {key} in self.data_buffer.keys()"
        return self.data_buffer[key][-1]

    def get_curiosity_data(self):
        '''
        返回用于好奇心机制的数据
        '''

        # T * [B, N] => [B, T, N] => [B*T, N]
        def func(x): return np.stack(x, axis=1).reshape(self.n_agents * self.eps_len, -1)

        data = {}
        for k in BatchExperiences._fields:
            assert k in self.data_buffer.keys(), f"assert {k} in self.data_buffer.keys()"
            if isinstance(self.data_buffer[k][0], tuple):
                data[k] = NamedTupleStaticClass.pack(self.data_buffer[k], func=func)
            else:
                data[k] = func(self.data_buffer[k])
        return BatchExperiences(**data)

    def update_reward(self, r: np.ndarray):
        '''
        r: [B*T, N]
        '''
        r = r.reshape(self.n_agents, self.eps_len, -1)
        for i in range(self.eps_len):
            self.data_buffer['reward'][i] += r[:, i]

    def convert_action2one_hot(self, a_counts):
        '''
        用于在训练前将buffer中的离散动作的索引转换为one_hot类型
        '''
        assert 'action' in self.data_buffer.keys(), "assert 'action' in self.data_buffer.keys()"
        self.data_buffer['action'] = [int2one_hot(a.astype(np.int32), a_counts) for a in self.data_buffer['action']]

    def normalize_vector_obs(self, func):
        '''
        TODO: Annotation
        '''
        assert 'obs' in self.data_buffer.keys(), "assert 'obs' in self.data_buffer.keys()"
        assert 'obs_' in self.data_buffer.keys(), "assert 'obs_' in self.data_buffer.keys()"
        self.data_buffer['obs'] = [NamedTupleStaticClass.data_convert(func, obs, keys=['vector']) for obs in self.data_buffer['obs']]
        self.data_buffer['obs_'] = [NamedTupleStaticClass.data_convert(func, obs_, keys=['vector']) for obs_ in self.data_buffer['obs_']]

    def sample_generater(self, batch_size: int = None):
        '''
        create sampling data iterator without using rnn.

        params:
            batch_size: the batch size of training data
            keys: the keys of data that should be sampled to train policies
        return:
            sampled data.
        '''

        batch_size = batch_size or self.batch_size

        buffer = {}
        # T * [B, N] => [T*B, N]
        for k in self.sample_data_type._fields:
            assert k in self.data_buffer.keys(), f"assert {k} in self.data_buffer.keys()"
            if isinstance(self.data_buffer[k][0], tuple):
                buffer[k] = NamedTupleStaticClass.pack(self.data_buffer[k], func=np.concatenate)
            else:
                buffer[k] = np.concatenate(self.data_buffer[k])

        idxs = np.arange(self.eps_len * self.n_agents)
        np.random.shuffle(idxs)
        for i in range(0, self.eps_len * self.n_agents, batch_size * self.n_agents):
            _idxs = idxs[i:i + batch_size * self.n_agents]
            data = []
            for k in self.sample_data_type._fields:
                if isinstance(buffer[k], tuple):
                    data.append(NamedTupleStaticClass.getbatchitems(buffer[k], _idxs))
                else:
                    data.append(buffer[k][_idxs])
            yield self.sample_data_type._make(data), (None, )

    def sample_generater_rnn(self, batch_size: int = None, rnn_time_step: int = None):
        '''
        create rnn sampling data iterator.

        params:
            rnn_time_step: the length of time slide window
        return:
            sampled data.
        '''
        batch_size = batch_size or self.batch_size
        rnn_time_step = rnn_time_step or self.rnn_time_step

        # TODO: 未done导致的episode切换需要严谨处理
        # T * [B, 1] => [T, B] => [B, T]
        done = np.asarray(self.data_buffer['done']).squeeze().transpose((1, 0))
        B, T = done.shape
        done_dict = defaultdict(list)
        for i, j in zip(*np.where(done)):
            done_dict[i].append(j)

        available_sample_range = defaultdict(list)
        count = 0   # 记录不交叉分割，最多有几段
        for i in range(B):
            idxs = [-1] + done_dict[i] + [T - 1]
            for x, y in zip(idxs[:-1], idxs[1:]):
                if y - rnn_time_step + 1 > x:
                    available_sample_range[i].append([x + 1, y - rnn_time_step + 1])    # 左开右开
                    count += (y - x) // 2

        # prevent total_eps_num is smaller than batch_size
        while batch_size > count:
            batch_size //= 2

        for _ in range(count // batch_size):
            samples = []
            sample_cs = []
            for i in range(batch_size):  # B
                batch_idx = random.choice(list(available_sample_range.keys()))
                sample_range = random.choice(available_sample_range[batch_idx])
                time_idx = random.randint(*sample_range)

                sample_exp = {}
                for k in self.sample_data_type._fields:
                    assert k in self.data_buffer.keys(), f"assert {k} in self.data_buffer.keys()"
                    d = self.data_buffer[k][time_idx:time_idx + rnn_time_step]    # T * [B, N]
                    if isinstance(self.data_buffer[k][0], tuple):
                        d = [NamedTupleStaticClass.getitem(_d, batch_idx) for _d in d]
                        sample_exp[k] = NamedTupleStaticClass.pack(d)   # [T, N]
                    else:
                        d = [_d[batch_idx] for _d in d]
                        sample_exp[k] = np.asarray(d)
                samples.append(self.sample_data_type(**sample_exp))

                sample_cs.append((cs[time_idx][batch_idx] for cs in self.cell_state_buffer))
            cs = tuple(np.asarray(x) for x in zip(*sample_cs))   # [B, N]
            yield NamedTupleStaticClass.pack(samples, func=np.concatenate), cs    # [B*T, N]

    def clear(self):
        '''
        清空临时存储经验池
        '''
        self.eps_len = 0
        for k in self.data_buffer.keys():
            self.data_buffer[k].clear()
        for l in self.cell_state_buffer:
            l.clear()

    def __getattr__(self, name):
        '''
        TODO: Annotation
        '''
        return self.data_buffer[name]

    def __getitem__(self, name):
        '''
        TODO: Annotation
        '''
        return self.data_buffer[name]

    def __str__(self):
        return str(self.data_buffer)


if __name__ == "__main__":
    pass
