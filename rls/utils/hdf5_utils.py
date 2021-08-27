import h5py
import numpy as np

from typing import NamedTuple

# TODO:


def namedtuple2hdf5(path: str, data: NamedTuple):
    def save(hf, data):
        for k, v in data._asdict().items():
            if isinstance(v, tuple):
                hfk = hf.create_group(k)
                save(hfk, v)
            else:
                hf.create_dataset(k, data=v)
    with h5py.File(path, 'w') as hf:
        save(hf, data)


def hdf52namedtuple(path: str, data_type: type):
    def load(hf, data_type):
        x = {}
        for k, v in hf.items():
            if isinstance(v, h5py.Group):
                x[k] = load(v, data_type._field_types.get(k))
            else:
                x[k] = v[:]
        return data_type(**x)
    with h5py.File(path, 'r') as hf:
        data = load(hf, data_type)
    return data


def hdf52dict(path: str):
    def load(hf):
        x = {}
        for k, v in hf.items():
            if isinstance(v, h5py.Group):
                x[k] = load(v)
            else:
                x[k] = v[:]
        return x
    with h5py.File(path, 'r') as hf:
        data = load(hf)
    return data


if __name__ == '__main__':

    class Test2(NamedTuple):
        c: np.ndarray

    class Test(NamedTuple):
        a: Test2
        b: np.ndarray

    data_to_write = Test(a=Test2(c=np.random.random(size=(100, 100))),
                         b=np.random.random(size=(100, 100)))
    namedtuple2hdf5('name-of-file.h5', data_to_write)

    data = hdf52namedtuple('name-of-file.h5', Test)

    data = hdf52dict('name-of-file.h5')
