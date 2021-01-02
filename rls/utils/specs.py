import numpy as np

from enum import Enum
from typing import (Dict,
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

class ModelObservations(NamedTuple):
    '''
        agent's observation
    '''
    vector: np.ndarray
    visual: np.ndarray

class Experience(NamedTuple):
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
