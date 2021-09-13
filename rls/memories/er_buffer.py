from typing import Dict, List, NoReturn, Union

import numpy as np

from rls.common.specs import Data
from rls.common.when import Once

'''
batch like : {
            'agent_0': {
                'obs': {
                    'vector: {
                        'vector_0': np.ndarray,
                        'vector_1': np.ndarray,
                        ...
                    },
                    'visual: {
                        'visual_0': np.ndarray,
                        'visual_1': np.ndarray,
                        ...
                    }
                }
                'action': np.ndarray,
                'reward': np.ndarray,
                'done': np.ndarray,
                'rnncs_s': (np.ndarray, ) or [np.ndarray, ] or np.ndarray
            }
            ...
            'global':{
                'begin_mask': np.ndarray,
                ...
                'state': {
                    'state_0': np.ndarray,
                    'state_1': np.ndarray,
                    ...
                }
            }
        }
'''


class DataBuffer:

    def __init__(self,
                 n_copys=1,
                 batch_size=1,
                 buffer_size=4,
                 chunk_length=1):
        self.n_copys = n_copys
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self._chunk_length = chunk_length
        self.max_horizon = buffer_size // n_copys

        # [N, T, B, *]
        self._buffer = dict()   # {str: Union[Dict[str, Data], Data]}
        self._horizon_length = 0
        self._pointer = 0
        self._not_builded = Once()

    def keys(self):
        return self._buffer.keys()

    def _build_buffer(self, data):
        for k, v in data.items():
            if k not in self._buffer.keys():
                self._buffer[k] = {}
            for _k, _v in v.nested_dict().items():
                if _k not in self._buffer[k].keys():
                    self._buffer[k][_k] = np.empty(
                        (self.max_horizon,)+_v.shape, _v.dtype)

    def add(self, data: Dict[str, Data]):
        assert isinstance(data, dict), "assert isinstance(data, dict)"
        if self._not_builded():
            self._build_buffer(data)

        for k, v in data.items():
            for _k, _v in v.nested_dict().items():
                self._buffer[k][_k][self._pointer] = _v

        self._pointer = (self._pointer + 1) % self.max_horizon
        self._horizon_length = min(self._horizon_length+1, self.max_horizon)

    def sample(self, batchsize=None, chunk_length=None):
        if batchsize == 0:
            return self.all_data()

        B = batchsize or self.batch_size
        T = chunk_length or self._chunk_length
        assert T <= self._horizon_length

        if self._horizon_length == self.max_horizon:
            start = self._pointer - self.max_horizon
        else:
            start = 0
        end = self._pointer - T + 1

        x = np.random.randint(start, end, B)    # [B, ]
        y = np.random.randint(0, self.n_copys, B)  # (B, )
        # (T, B) + (B, ) = (T, B)
        xs = (np.tile(np.arange(T)[:, np.newaxis],
              B) + x) % self._horizon_length
        sample_idxs = (xs, y)
        samples = {}
        for k, v in self._buffer.items():
            samples[k] = Data.from_nested_dict(
                {_k: _v[sample_idxs] for _k, _v in v.items()}
            )
        return samples  # [T, B, *]

    def __repr__(self):
        str = ''
        for k, v in self._buffer.items():
            str += f'{k}:'
            if isinstance(v, dict):
                for _k, _v in v.items():
                    str += f'\n  {_k}:{_v[:self._horizon_length]}'
            else:
                str += f'  {v[:self._horizon_length]}'
            str += '\n'
        return str

    @property
    def can_sample(self):
        return (self._horizon_length - self._chunk_length) * self.n_copys >= self.batch_size

    @property
    def is_multi(self) -> bool:
        return len(self._buffer.keys()) > 2

    def __getitem__(self, item):
        return self._buffer[item]

    def all_data(self):
        samples = {}
        for k, v in self._buffer.items():
            samples[k] = Data.from_nested_dict(v)
        return samples

    def clear(self):
        self._horizon_length = 0
        self._pointer = 0
