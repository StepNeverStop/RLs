
import numpy as np
import torch as t

from enum import Enum
from typing import (Dict,
                    List,
                    Union,
                    Tuple,
                    Iterator,
                    Callable)
from dataclasses import (dataclass,
                         astuple,
                         asdict,
                         is_dataclass,
                         make_dataclass)

from rls.utils.sundry_utils import nested_tuple


@dataclass(frozen=True)
class ObsSpec:
    vector_dims: List[int]
    visual_dims: List[Union[List[int], Tuple[int]]]

    @property
    def total_vector_dim(self):
        return sum(self.vector_dims)

    @property
    def has_vector_observation(self):
        return len(self.vector_dims) > 0

    @property
    def has_visual_observation(self):
        return len(self.visual_dims) > 0

    @staticmethod
    def construct_same_concat(obs_spec, n: int):
        # TODO: 优化，检查所有维度是否匹配
        return ObsSpec(
            vector_dims=[vec_dim * n for vec_dim in obs_spec.vector_dims],
            visual_dims=[[vis_dim[0], vis_dim[1], vis_dim[-1] * n] if len(vis_dim) == 3 else [] for vis_dim in obs_spec.visual_dims],
        )


@dataclass(frozen=True)
class EnvGroupArgs:
    obs_spec: ObsSpec
    a_dim: int
    is_continuous: bool
    n_copys: int


@dataclass
class Data:

    @property
    def tensor(self):
        params = {}
        for k, v in self.__dict__.items():
            params[k] = v.tensor if isinstance(v, Data) else t.as_tensor(v).float()
        return self.__class__(**params)

    @property
    def numpy(self):
        params = {}
        for k, v in self.__dict__.items():
            params[k] = v.numpy if isinstance(v, Data) else np.asarray(v)
        return self.__class__(**params)

    def asdict(self, recursive=False):
        return asdict(self) if recursive else self.__dict__

    def astuple(self, recursive=False):
        return astuple(self) if recursive else self.__dict__.values()

    def __getitem__(self, i):
        params = {}
        for k, v in self.asdict().items():
            params[k] = v[i]
        return self.__class__(**params)

    def __setitem__(self, i, v):
        '''TODO: Test'''
        for k in self.__dict__.keys():
            getattr(self, k)[i] = v[k]

    def __len__(self):
        for k, v in self.__dict__.items():
            return len(v) if isinstance(v, Data) else v.shape[0]

    def __eq__(self, other):
        '''TODO: Annotation'''
        for s, o in zip(nested_tuple(self.astuple(recursive=True)), nested_tuple(other.astuple(recursive=True))):
            if not np.allclose(s, o, equal_nan=True):
                return False
        else:
            return True

    def convert_(self, func: Callable, keys=None):
        keys = keys or self.__dict__.keys()
        for k in keys:
            v = getattr(self, k)
            if isinstance(v, Data):
                v.convert_(func)
            else:
                setattr(self, k, func(v))

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


def generate_obs_dataformat(n_copys, item_nums, name='obs'):
    if item_nums == 0:
        return lambda *args: make_dataclass('dataclass', [name], bases=(Data,))(np.full((n_copys, 0), [], dtype=np.float32))
    else:
        return make_dataclass('dataclass', [name+f'_{str(i)}' for i in range(item_nums)], bases=(Data,))


@dataclass(eq=False)
class ModelObservations(Data):
    '''
        agent's observation
    '''
    vector: Data
    visual: Data

    def flatten_vector(self):
        '''
        TODO: Annotation
        '''
        func = np.hstack if isinstance(self.first_vector(), np.ndarray) \
            else lambda x: t.cat(x, -1)
        return func(self.vector.astuple(recursive=True))

    def first_vector(self):
        '''
        TODO: Annotation
        '''
        return self.vector.astuple(recursive=True)[0]

    def first_visual(self):
        '''
        TODO: Annotation
        '''
        return self.visual.astuple(recursive=True)[0]


@dataclass(eq=False)
class BatchExperiences(Data):
    '''
        format of experience that needed to be stored in replay buffer.
    '''
    obs: ModelObservations
    action: np.ndarray
    reward: np.ndarray
    obs_: ModelObservations
    done: np.ndarray


@dataclass
class SingleModelInformation:
    '''
        format of information received after each environment timestep
    '''
    corrected_obs: ModelObservations
    obs: ModelObservations
    reward: np.ndarray
    done: np.ndarray
    info: Dict


class GymVectorizedType(Enum):
    RAY = 'ray'
    VECTOR = 'vector'
    MULTITHREADING = 'multithreading'
    MULTIPROCESSING = 'multiprocessing'
