import numpy as np

from enum import Enum
from typing import (Dict,
                    List,
                    Iterator,
                    Callable,
                    NamedTuple)
from collections import namedtuple

SingleAgentEnvArgs = namedtuple('SingleAgentEnvArgs',
                                [
                                    's_dim',
                                    'visual_sources',
                                    'visual_resolutions',
                                    'a_dim',
                                    'is_continuous',
                                    'n_agents'
                                ])

MultiAgentEnvArgs = namedtuple('MultiAgentEnvArgs',
                               SingleAgentEnvArgs._fields + ('behavior_controls',))

UnitySingleBehaviorInfo = namedtuple('UnitySingleBehaviorInfo',
                                     [
                                         'behavior_name',
                                         'n_agents_control',
                                         'is_continuous'
                                     ])


class NamedTupleStaticClass:

    @staticmethod
    def len(nt: NamedTuple) -> int:
        '''
        计算namedtuple的元素个数
        '''
        for data in nt:
            if isinstance(data, tuple):
                return NamedTupleStaticClass.len(data)
            elif isinstance(data, np.ndarray):
                return data.shape[0]
            elif isinstance(data, list):
                return len(data)
        else:
            raise ValueError(f"cannot compute length of {nt.__class__}.")

    @staticmethod
    def getitem(nt: NamedTuple, i: int) -> NamedTuple:
        if isinstance(nt, tuple):
            x = []
            for data in nt:
                x.append(NamedTupleStaticClass.getitem(data, i))
            return nt.__class__._make(x)
        else:
            return nt[i]

    @staticmethod
    def getbatchitems(nt: NamedTuple, idxs: np.ndarray) -> NamedTuple:
        if isinstance(nt, tuple):
            x = []
            for data in nt:
                x.append(NamedTupleStaticClass.getbatchitems(data, idxs))
            return nt.__class__._make(x)
        else:
            return nt[idxs]

    @staticmethod
    def unpack(nt: NamedTuple) -> Iterator[NamedTuple]:
        for i in range(NamedTupleStaticClass.len(nt)):
            yield NamedTupleStaticClass.getitem(nt, i)

    @staticmethod
    def pack(nts: List[NamedTuple], func: Callable = None) -> NamedTuple:
        x = []
        for datas in zip(*nts):
            if isinstance(datas[0], tuple):
                x.append(NamedTupleStaticClass.pack(datas, func))
            else:
                if func:
                    x.append(func(datas))
                else:
                    x.append(np.asarray(datas))
        return nts[0].__class__._make(x)

    @staticmethod
    def check_equal(x: NamedTuple, y: NamedTuple, k: str = None):
        def _check(d1, d2):
            if isinstance(d1, tuple) and isinstance(d2, tuple):
                return NamedTupleStaticClass.check_equal(d1, d2)
            elif isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray):
                return (d1 == d2).all()
            else:
                return False

        if k is not None:
            return _check(d1=getattr(x, k), d2=getattr(y, k))
        else:
            return all([_check(d1=getattr(x, k), d2=getattr(y, k)) for k in x._fields])

    @staticmethod
    def data_convert(func, nt, keys=None):
        if keys is None:
            x = []
            for data in nt:
                if isinstance(data, tuple):
                    x.append(NamedTupleStaticClass.data_convert(func, data))
                else:
                    x.append(func(data))
            return nt.__class__._make(x)
        else:
            x = {}
            for k in keys:
                data = getattr(nt, k)
                if isinstance(data, tuple):
                    x[k] = NamedTupleStaticClass.data_convert(func, data)
                else:
                    x[k] = func(data)
            return nt._replace(**x)

    @staticmethod
    def show_shape(nt):
        for k, v in nt._asdict().items():
            if isinstance(v, tuple):
                NamedTupleStaticClass.show_shape(v)
            else:
                print(k, v.shape)

    @staticmethod
    def check_len(nt: NamedTuple, l: int):
        ret = []
        for data in nt:
            if isinstance(data, tuple):
                ret.append(NamedTupleStaticClass.check_len(data, l))
            else:
                ret.append(data.shape[0] == l)
        return all(ret)


class ModelObservations(NamedTuple):
    '''
        agent's observation
    '''
    vector: np.ndarray
    visual: np.ndarray


class BatchExperiences(NamedTuple):
    '''
        format of experience that needed to be stored in replay buffer.
    '''
    obs: ModelObservations
    action: np.ndarray
    reward: np.ndarray
    obs_: ModelObservations
    done: np.ndarray


class SingleModelInformation(NamedTuple):
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


class VisualNetworkType(Enum):
    MATCH3 = 'match3'
    SIMPLE = 'simple'
    NATURE = 'nature'
    RESNET = 'resnet'
    DEEPCONV = 'deepconv'


class DefaultActivationFuncType(Enum):
    TANH = 'tanh'
    RELU = 'relu'
    ELU = 'elu'
    SWISH = 'swish'
    MISH = 'mish'


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
    ACTOR_CRITIC_VALUE_DET = 'ActorCriticValueDct'
    C51_DISTRIBUTIONAL = 'C51Distributional'
    QRDQN_DISTRIBUTIONAL = 'QrdqnDistributional'
    RAINBOW_DUELING = 'RainbowDueling'
    IQN_NET = 'IqnNet'


class MemoryNetworkType(Enum):
    GRU = 'gru'
    LSTM = 'lstm'
