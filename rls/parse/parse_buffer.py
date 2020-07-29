#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rls.parse.parse_buffer")
import importlib

from typing import \
    Optional

from rls.common.config import Config
from rls.memories.replay_buffer import ReplayBuffer

BufferDict = {
    'ER': 'ExperienceReplay',
    'PER': 'PrioritizedExperienceReplay',
    'NstepER': 'NStepExperienceReplay',
    'NstepPER': 'NStepPrioritizedExperienceReplay',
    'EpisodeER': 'EpisodeExperienceReplay'
}

def get_buffer(buffer_args: Config) -> Optional[ReplayBuffer]:
    '''
    parsing arguments of replay buffer
    params:
        buffer_args: configurations of replay buffer
    return:
        Correct experience replay mechanism.
        For On-Policy algorithms, they don't have to specify a replay buffer out of model class, so return None
    '''

    if buffer_args.get('buffer_size', 0) <= 0:
        logger.info('This algorithm does not need sepecify a data buffer oustside the model.')
        return None

    _buffer_type = buffer_args.get('type', 'None')
    logger.info(_buffer_type)
    
    if _buffer_type in BufferDict.keys():
        Buffer = getattr(importlib.import_module(f'rls.memories.replay_buffer'), 
                        BufferDict[_buffer_type])
        return Buffer(batch_size=buffer_args['batch_size'], 
                    capacity=buffer_args['buffer_size'], 
                    **buffer_args[_buffer_type].to_dict)
    else:
        return None