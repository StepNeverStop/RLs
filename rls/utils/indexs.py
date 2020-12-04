from enum import Enum
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
                               SingleAgentEnvArgs._fields + ('group_controls',))

UnitySingleAgentReturn = namedtuple('UnitySingleAgentReturn',
                                    [
                                        'vector',
                                        'visual',
                                        'reward',
                                        'done',
                                        'corrected_vector',
                                        'corrected_visual',
                                        'info'
                                    ])


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
