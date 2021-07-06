
import numpy as np
import tensorflow as tf

from enum import Enum
from typing import (Dict,
                    List,
                    Union,
                    Tuple,
                    Iterator,
                    Callable)
from dataclasses import (dataclass,
                         make_dataclass)
from collections import namedtuple


@dataclass
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


@dataclass
class EnvGroupArgs:
    obs_spec: ObsSpec
    a_dim: int
    is_continuous: bool
    n_copys: int


@dataclass
class RlsDataClass:

    def map_fn(self, func, keys=None):
        keys = keys or self.__dict__.keys()
        for k in keys:
            v = getattr(self, k)
            if isinstance(v, RlsDataClass):
                v.map_fn(func)
            else:
                setattr(self, k, func(v))

    def getitem(self, idx):
        params = {}
        for k, v in self.__dict__.items():
            if isinstance(v, RlsDataClass):
                params[k] = v.getitem(idx)
            else:
                params[k] = v[idx]
        return self.__class__(**params)

    def getbatchitems(self, idxs: np.ndarray):
        params = {}
        for k, v in self.__dict__.items():
            if isinstance(v, RlsDataClass):
                params[k] = v.getitem(idxs)
            else:
                params[k] = v[idxs]
        return self.__class__(**params)

    @property
    def batch_size(self):
        for k, v in self.__dict__.items():
            if isinstance(v, RlsDataClass):
                return v.batch_size
            else:
                return v.shape[0]

    def unpack(self) -> Iterator:
        '''
        TODO: Annotation
        '''
        for i in range(self.batch_size):
            yield self.getitem(i)

    @staticmethod
    def pack(rdss: List, func: Callable = None):
        '''
        TODO: Annotation
        '''
        params = {}
        for k, v in rdss[0].__dict__.items():
            d = [getattr(rds, k) for rds in rdss]
            if isinstance(v, RlsDataClass):
                params[k] = RlsDataClass.pack(d, func)
            else:
                params[k] = func(d) if func else np.asarray(d)
        return rdss[0].__class__(**params)

    @staticmethod
    def check_equal(x, y, k: str = None):
        '''
        TODO: Annotation
        '''
        def _check(d1, d2):
            if isinstance(d1, RlsDataClass) and isinstance(d2, RlsDataClass):
                return RlsDataClass.check_equal(d1, d2)
            elif isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray):
                return np.allclose(d1, d2, equal_nan=True)
            else:
                return False

        if k is not None:
            return _check(d1=getattr(x, k), d2=getattr(y, k))
        else:
            return all([_check(d1=getattr(x, k), d2=getattr(y, k)) for k in x.__dict__.keys()])

    def check_len(self, l: int):
        '''
        TODO: Annotation
        '''
        ret = []
        for v in self.values():
            if isinstance(v, RlsDataClass):
                ret.append(RlsDataClass.check_len(v, l))
            else:
                ret.append(v.shape[0] == l)
        return all(ret)

    @property
    def nt(self):
        params = {}
        nt = namedtuple('nt', self.__dict__.keys())
        for k, v in self.__dict__.items():
            if isinstance(v, RlsDataClass):
                params[k] = v.nt
            else:
                params[k] = v
        return nt(**params)


def generate_obs_dataformat(n_copys, item_nums, name='obs'):
    if item_nums == 0:
        return lambda *args: make_dataclass('dataclass', [name], bases=(RlsDataClass,))(np.full((n_copys, 0), [], dtype=np.float32))
    else:
        return make_dataclass('dataclass', [name+f'_{str(i)}' for i in range(item_nums)], bases=(RlsDataClass,))


@dataclass
class ModelObservations(RlsDataClass):
    '''
        agent's observation
    '''
    vector: RlsDataClass
    visual: RlsDataClass

    def flatten_vector(self):
        '''
        TODO: Annotation
        '''
        func = np.hstack if isinstance(self.first_vector(), np.ndarray) \
            else lambda x: tf.concat(x, axis=-1)
        return func(self.vector.values())

    def first_vector(self):
        '''
        TODO: Annotation
        '''
        return self.vector.values()[0]

    def first_visual(self):
        '''
        TODO: Annotation
        '''
        return self.visual.values()[0]


@dataclass
class BatchExperiences(RlsDataClass):
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


class VectorNetworkType(Enum):
    CONCAT = 'concat'
    ADAPTIVE = 'adaptive'


class VisualNetworkType(Enum):
    MATCH3 = 'match3'
    SIMPLE = 'simple'
    NATURE = 'nature'
    RESNET = 'resnet'
    DEEPCONV = 'deepconv'


class MemoryNetworkType(Enum):
    GRU = 'gru'
    LSTM = 'lstm'


class DefaultActivationFuncType(Enum):
    TANH = 'tanh'
    RELU = 'relu'
    ELU = 'elu'
    SWISH = 'swish'  # https://arxiv.org/abs/1710.05941
    MISH = 'mish'   # https://arxiv.org/abs/1908.08681


class OutputNetworkType(Enum):
    ACTOR_DPG = 'ActorDPG'
    ACTOR_MU = 'ActorMu'
    ACTOR_MU_LOGSTD = 'ActorMuLogstd'
    ACTOR_CTS = 'ActorCts'
    ACTOR_DCT = 'ActorDct'
    CRITIC_QVALUE_ONE = 'CriticQvalueOne'
    CRITIC_QVALUE_ONE_DDPG = 'CriticQvalueOneDDPG'
    CRITIC_QVALUE_ONE_TD3 = 'CriticQvalueOneTD3'
    CRITIC_VALUE = 'CriticValue'
    CRITIC_QVALUE_ALL = 'CriticQvalueAll'
    CRITIC_QVALUE_BOOTSTRAP = 'CriticQvalueBootstrap'
    CRITIC_DUELING = 'CriticDueling'
    OC_INTRA_OPTION = 'OcIntraOption'
    AOC_SHARE = 'AocShare'
    PPOC_SHARE = 'PpocShare'
    ACTOR_CRITIC_VALUE_CTS = 'ActorCriticValueCts'
    ACTOR_CRITIC_VALUE_DCT = 'ActorCriticValueDct'
    C51_DISTRIBUTIONAL = 'C51Distributional'
    QRDQN_DISTRIBUTIONAL = 'QrdqnDistributional'
    RAINBOW_DUELING = 'RainbowDueling'
    IQN_NET = 'IqnNet'
