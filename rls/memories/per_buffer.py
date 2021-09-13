import sys
from typing import Dict, List, NoReturn, Union

import numpy as np

from rls.common.specs import Data
from rls.memories.er_buffer import DataBuffer
from rls.memories.sum_tree import Sum_Tree


class PrioritizedDataBuffer(DataBuffer):

    def __init__(self,
                 n_copys=1,
                 batch_size=1,
                 buffer_size=4,
                 chunk_length=1,
                 max_train_step: int = sys.maxsize,

                 alpha: float = 0.6,
                 beta: float = 0.4,
                 epsilon: float = 0.01,
                 global_v: bool = False):
        '''
            max_train_step: use for calculating the decay interval of beta
            alpha: control sampling rule, alpha -> 0 means uniform sampling, alpha -> 1 means complete td_error sampling
            beta: control importance sampling ratio, beta -> 0 means no IS, beta -> 1 means complete IS.
            epsilon: a small positive number that prevents td-error of 0 from never being replayed.
            global_v: whether using the global
        '''
        super().__init__(n_copys=n_copys,
                         batch_size=batch_size,
                         buffer_size=buffer_size,
                         chunk_length=chunk_length)
        # [T0B0, ..., T0BN, T1B0, ..., T1BN, ..., TNBN]
        self._tree = Sum_Tree(self.buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.beta_interval = (1. - beta) / max_train_step
        self.epsilon = epsilon
        self.global_v = global_v
        self.min_p = sys.maxsize
        self.max_p = np.power(self.epsilon, self.alpha)

    def add(self, data: Dict[str, Data]):
        super().add(data)
        self._tree.add_batch(np.full(self.n_copys, self.max_p),
                             n_step_delay=self._chunk_length-1)

    def sample(self, batchsize=None, chunk_length=None):
        B = batchsize or self.batch_size
        if chunk_length is not None:     # TODO: optimize chunk_length
            T = min(chunk_length, self._chunk_length)
        else:
            T = self._chunk_length
        assert T <= self._horizon_length

        all_intervals = np.linspace(0, self._tree.total, B + 1)
        ps = np.random.uniform(all_intervals[:-1], all_intervals[1:])
        didx, p = self._tree.get_batch_parallel(ps)
        self.last_indexs = didx
        _min_p = self.min_p if self.global_v and self.min_p < sys.maxsize else p.min()
        x, y = didx // self.n_copys, didx % self.n_copys    # t, b

        # (T, B) + (B, ) = (T, B)
        xs = (np.tile(np.arange(T)[:, np.newaxis],
              B) + x) % self._horizon_length
        sample_idxs = (xs, y)

        # weights of variables by using Importance Sampling
        isw = np.power(_min_p / p, self.beta).reshape(B, -1)
        samples = {}
        for k, v in self._buffer.items():
            samples[k] = Data.from_nested_dict(
                {_k: _v[sample_idxs] for _k, _v in v.items()}
            )
        if self.is_multi:   # TODO: optimize and check
            samples['global'].update(isw=isw)
        else:
            for k, v in self._buffer.items():
                samples[k].update(isw=isw)
        return samples

    def update(self, priorities: np.ndarray) -> NoReturn:
        '''
        params: 
            priorities: [T, B, 1]
        '''
        if priorities.ndim == 3:
            priorities = priorities[0]  # [B, 1]
        priorities = np.abs(np.ravel(priorities))
        priorities = np.power(priorities + self.epsilon, self.alpha)
        self.beta = min(self.beta + self.beta_interval, 1.)
        self.min_p = min(self.min_p, priorities.min())
        self.max_p = max(self.max_p, priorities.max())
        self._tree.update_batch(self.last_indexs, priorities)
