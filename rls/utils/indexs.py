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


class VisualEncoderType(Enum):
    MATCH3 = 'match3'
    SIMPLE = 'simple'
    NATURE = 'nature'

class DefaultActivationFuncType(Enum):
    TANH = 'tanh'
    RELU = 'relu'
    ELU = 'elu'
    SWISH = 'swish'
    MISH = 'mish'