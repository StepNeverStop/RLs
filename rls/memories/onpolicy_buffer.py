import sys
from typing import Dict, List, NoReturn, Union

import numpy as np

from rls.common.specs import Data
from rls.memories.er_buffer import DataBuffer
from rls.memories.sum_tree import Sum_Tree


class OnPolicyDataBuffer(DataBuffer):

    def __init__(self,
                 n_copys=1,
                 batch_size=1,
                 buffer_size=4,
                 chunk_length=1):
        super().__init__(n_copys=n_copys,
                         batch_size=batch_size,
                         buffer_size=buffer_size,
                         chunk_length=chunk_length)

    def all_data(self):
        samples = {}
        for k, v in self._buffer.items():
            samples[k] = Data.from_nested_dict(v)
        return samples

    @property
    def can_sample(self):
        return (self._horizon_length // self.chunk_length) * self.n_copys >= self.batch_size

    def clear(self):
        self._horizon_length = 0
        self._pointer = 0
