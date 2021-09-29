from typing import Dict, Union

import numpy as np


def get_first_vector(d: "Data"):
    assert hasattr(d, 'vector'), "assert hasattr(d, 'vector')"
    return d.vector.vector_0


def get_first_visual(d: "Data"):
    assert hasattr(d, 'visual'), "assert hasattr(d, 'visual')"
    return d.visual.visual_0


def get_flat_vector(d: "Data"):
    assert hasattr(d, 'vector'), "assert hasattr(d, 'vector')"
    return np.concatenate(d.vector.values(), axis=-1)


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
                v.convert_(func)  # TODO: optimize
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

    @property
    def shape(self):
        for k, v in self.__dict__.items():
            return v.shape

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
        """TODO: Annotation"""
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
                x.update(v.nested_dict(pre=pre + f'{k}{mark}', mark=mark))
            else:
                x[pre + k] = v
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

    def sample(self, _t, _b, repeat=True):
        for idxs in self._yield_sample_indexs(_t, _b, repeat):
            yield self[idxs]

    def _yield_sample_indexs(self, _t, _b, repeat=True):
        T, B = self.shape[:2]
        if repeat:
            for _ in range((T - _t + 1) * B // _b):
                x = np.random.randint(0, T - _t + 1, _b)  # [B, ]
                y = np.random.randint(0, B, _b)  # (B, )
                xs = np.tile(np.arange(_t)[:, np.newaxis], _b) + x  # (T, B) + (B, ) = (T, B)
                sample_idxs = (xs, y)
                yield sample_idxs
        else:
            # [N, ] + [B, 1] => [B, N]
            x = np.arange(0, T - _t + 1, _t) + np.random.randint(0, T % _t + 1, B)[:, np.newaxis]
            y = np.arange(B).repeat(x.shape[-1])  # [B*N]
            x = x.ravel()  # [B*N]
            idxs = np.arange(len(x))  # [B*N]
            np.random.shuffle(idxs)  # [B*N]
            for i in range(len(idxs) // _b):
                # [T, B]
                start, end = i * _b, (i + 1) * _b
                xs = x[start:end] + np.tile(np.arange(_t)[:, np.newaxis], _b)
                sample_idxs = (xs, y[start:end])
                yield sample_idxs


class DictCls(dict):

    def __getattr__(self, name):
        assert name not in self.keys(), 'assert name not in self.keys()'
        return [v.get(name) if isinstance(v, dict) else getattr(v, name) for k, v in self.items()]
