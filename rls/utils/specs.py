import numpy as np
import tensorflow as tf

from enum import Enum
from typing import (Dict,
                    List,
                    Union,
                    Tuple,
                    Iterator,
                    Callable,
                    NamedTuple)
from collections import namedtuple


class ObsSpec(NamedTuple):
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


EnvGroupArgs = NamedTuple('EnvGroupArgs',
                          [
                              ('obs_spec', ObsSpec),
                              ('a_dim', int),
                              ('is_continuous', bool),
                              ('n_copys', int)
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
        '''
        TODO: Annotation
        '''
        if isinstance(nt, tuple):
            x = []
            for data in nt:
                x.append(NamedTupleStaticClass.getitem(data, i))
            return nt.__class__._make(x)
        else:
            return nt[i]

    @staticmethod
    def getbatchitems(nt: NamedTuple, idxs: np.ndarray) -> NamedTuple:
        '''
        TODO: Annotation
        '''
        if isinstance(nt, tuple):
            x = []
            for data in nt:
                x.append(NamedTupleStaticClass.getbatchitems(data, idxs))
            return nt.__class__._make(x)
        else:
            return nt[idxs]

    @staticmethod
    def unpack(nt: NamedTuple) -> Iterator[NamedTuple]:
        '''
        TODO: Annotation
        '''
        for i in range(NamedTupleStaticClass.len(nt)):
            yield NamedTupleStaticClass.getitem(nt, i)

    @staticmethod
    def pack(nts: List[NamedTuple], func: Callable = None) -> NamedTuple:
        '''
        TODO: Annotation
        '''
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
        '''
        TODO: Annotation
        '''
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
        '''
        TODO: Annotation
        '''
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
        '''
        TODO: Annotation
        '''
        # TODO: 优化显示
        for k, v in nt._asdict().items():
            if isinstance(v, tuple):
                NamedTupleStaticClass.show_shape(v)
            else:
                print(k, v.shape)

    @staticmethod
    def check_len(nt: NamedTuple, l: int):
        '''
        TODO: Annotation
        '''
        ret = []
        for data in nt:
            if isinstance(data, tuple):
                ret.append(NamedTupleStaticClass.check_len(data, l))
            else:
                ret.append(data.shape[0] == l)
        return all(ret)

    @staticmethod
    def union(nt: Union[np.ndarray, NamedTuple], func: Callable = None):
        '''
        TODO: Annotation
        '''
        if isinstance(nt, tuple):
            x = [NamedTupleStaticClass.union(data, func) for data in nt]
            return func(x)
        else:
            return nt

    @staticmethod
    def generate_obs_namedtuple(n_copys, item_nums, name='namedtuple'):
        '''
        TODO: 待优化删除
        '''
        if item_nums == 0:
            return lambda *args, **kwargs: NamedTuple('obs_namedtuple', [(f'{name}', np.ndarray)])(np.full((n_copys, 0), [], dtype=np.float32))
        else:
            return NamedTuple('obs_namedtuple', [(f'{name}_{str(i)}', np.ndarray) for i in range(item_nums)])

    @staticmethod
    def dict2namedtuple(data: dict, data_type: type):
        x = {}
        for k, v in data.items():
            if isinstance(v, dict):
                x[k] = NamedTupleStaticClass.dict2namedtuple(v, data_type._field_types.get(k))
            else:
                x[k] = v
        return data_type(**x)


class ModelObservations(NamedTuple):
    '''
        agent's observation
    '''
    vector: NamedTuple  # NamedTupleStaticClass.generate_obs_namedtuple
    visual: NamedTuple  # NamedTupleStaticClass.generate_obs_namedtuple

    def flatten_vector(self):
        '''
        TODO: Annotation
        '''
        func = np.hstack if isinstance(self.first_vector(), np.ndarray) else lambda x: tf.concat(x, axis=-1)
        return NamedTupleStaticClass.union(self.vector, func=func)

    def first_vector(self):
        '''
        TODO: Annotation
        '''
        return self.vector[0]

    def first_visual(self):
        '''
        TODO: Annotation
        '''
        return self.visual[0]

    @staticmethod
    def stack(obs: NamedTuple, obs_: NamedTuple):
        '''
        TODO: Annotation
        '''
        vector = [tf.concat([o, o_], axis=0) for o, o_ in zip(obs.vector, obs_.vector)]
        visual = [tf.concat([o, o_], axis=0) for o, o_ in zip(obs.visual, obs_.visual)]
        return ModelObservations(vector=obs.vector.__class__._make(vector),
                                 visual=obs.visual.__class__._make(visual))

    @staticmethod
    def stack_rnn(obs: NamedTuple, obs_: NamedTuple, episode_batch_size: int):
        '''
        TODO: Annotation
        '''
        # [B*T, N] => [B, T, N]
        _obs = NamedTupleStaticClass.data_convert(lambda x: tf.reshape(x, [episode_batch_size, -1, *x.shape[1:]]), obs)
        _obs_ = NamedTupleStaticClass.data_convert(lambda x: tf.reshape(x, [episode_batch_size, -1, *x.shape[1:]]), obs_)
        # TODO: 优化
        # [B, T, N], [B, T, N] => [B, T+1, N] => [B*(T+1), N]
        vector = [tf.reshape(tf.concat([o, o_[:, -1:]], axis=1), [-1, *o.shape[2:]]) for o, o_ in zip(_obs.vector, _obs_.vector)]
        visual = [tf.reshape(tf.concat([o, o_[:, -1:]], axis=1), [-1, *o.shape[2:]]) for o, o_ in zip(_obs.visual, _obs_.visual)]
        return ModelObservations(vector=obs.vector.__class__._make(vector),
                                 visual=obs.visual.__class__._make(visual))


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
