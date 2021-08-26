
import numpy as np
import torch as t

from typing import (Dict,
                    List,
                    Union,
                    Tuple,
                    Optional,
                    Iterator,
                    Callable)
from dataclasses import dataclass
from collections import defaultdict

# TODO:


class Every:

    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, step):
        if self._last is None:
            self._last = step
            return True
        if step >= self._last + self._every:
            self._last += self._every
            return True
        return False


class Once:

    def __init__(self):
        self._once = True

    def __call__(self):
        if self._once:
            self._once = False
            return True
        return False


@dataclass
class SensorSpec:
    vector_dims: Optional[List[int]] = None
    visual_dims: Optional[List[Union[List[int], Tuple[int]]]] = None
    other_dims: int = 0

    @property
    def total_vector_dim(self):
        '''TODO: Remove'''
        return sum(self.vector_dims)

    @property
    def has_vector_observation(self):
        return self.vector_dims is not None and len(self.vector_dims) > 0

    @property
    def has_visual_observation(self):
        return self.visual_dims is not None and len(self.visual_dims) > 0

    @property
    def has_other_observation(self):
        return self.other_dims > 0


@dataclass
class EnvAgentSpec:
    obs_spec: SensorSpec
    a_dim: int
    is_continuous: bool


class Data:

    def __init__(self, data: "Data" = None, **kwargs):
        self.update(data or kwargs)

    def update(self, data: Union["Data", Dict] = None, **kwargs):
        _data = data or kwargs
        for k, v in _data.items():
            if isinstance(v, dict):
                setattr(self, k, self.__class__(**v))
            else:
                setattr(self, k, v)

    def convert_(self, func, keys=None):
        for k in keys or self.keys():
            v = getattr(self, k)
            if isinstance(v, Data):
                v.convert_(func)    # TODO: optimize
            else:
                setattr(self, k, func(v))

    def convert(self, func, keys=None):
        params = {}
        for k in keys or self.keys():
            v = getattr(self, k)
            if isinstance(v, Data):
                params[k] = v.convert(func)
            else:
                params[k] = func(v)
        return self.__class__(**params)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __getitem__(self, item):
        params = {}
        for k in self.keys():
            params[k] = getattr(self, k)[item]
        return self.__class__(**params)

    def __setitem__(self, item, value):
        for k in self.keys():
            getattr(self, k)[item] = getattr(value, k)

    def __repr__(self):
        # TODO
        str = ''
        for k in self.keys():
            str += f'{k}:\n  {getattr(self, k)}\n'
        return str

    def __eq__(self, other):
        '''TODO: Annotation'''
        assert isinstance(other, Data), 'assert isinstance(other, Data)'
        for x, y in zip(self.values(), other.values()):
            if isinstance(x, Data) and isinstance(y, Data):
                return x == y
            elif isinstance(x, np.ndarray):
                return np.allclose(x, y, equal_nan=True)
        return True

    def __len__(self):
        for v in self.values():
            return len(v) if isinstance(v, Data) else v.shape[0]

    def nested_dict(self, pre='', mark='!'):
        x = dict()
        for k, v in self.items():
            if isinstance(v, Data):
                x.update(v.nested_dict(pre=pre+f'{k}{mark}', mark=mark))
            else:
                x[pre+k] = v
        return x

    @staticmethod
    def from_nested_dict(nested_dict, mark='!'):

        def func3(params, value, keys=[]):
            if keys[0] not in params.keys():
                params[keys[0]] = {}
            if len(keys) > 1:
                params.update({keys[0]: func3(params[keys[0]], value, keys[1:])})
            else:
                params.update({keys[0]: value})
            return params

        params = dict()
        for k, v in nested_dict.items():
            func3(params, v, k.split(mark))
        return __class__(**params)

    def to_dict(self):
        x = dict()
        for k, v in self.items():
            if isinstance(v, Data):
                x[k] = v.to_dict()
            else:
                x[k] = v
        return x

    def get(self, name, value=None):
        if name in self.keys():
            return getattr(self, name)
        else:
            return value

    # TODO: remove
    def unpack(self) -> Iterator:
        for i in range(len(self)):
            yield self[i]

    @staticmethod
    def pack(ds: List, func: Callable = lambda x: np.asarray(x)):
        '''
        TODO: Annotation
        '''
        params = {}
        for k, v in ds[0].__dict__.items():
            d = [getattr(rds, k) for rds in ds]
            params[k] = Data.pack(d, func) if isinstance(v, Data) else func(d)
        return ds[0].__class__(**params)
