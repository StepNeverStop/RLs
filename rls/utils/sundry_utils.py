#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
import random
import numpy as np
import tensorflow as tf

from typing import (List,
                    NoReturn)

from rls.utils.display import colorize
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


def check_or_create(dicpath: str, name: str = '') -> NoReturn:
    """
    check dictionary whether existing, if not then create it.
    """
    if not os.path.exists(dicpath):
        os.makedirs(dicpath)
        logger.info(colorize(
            ''.join([f'create {name} directionary :', dicpath]
                    ), color='green'))


def set_global_seeds(seed: int) -> NoReturn:
    """
    Set the random seed of tensorflow, numpy and random.
    params:
        seed: an integer refers to the random seed
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class LinearAnnealing:

    def __init__(self, x: float, x_: float, end: int):
        '''
        Params: 
            x: start value
            x_: end value
            end: annealing time
        '''
        assert end != 0, 'the time steps for annealing must larger than 0.'
        self.x = x
        self.x_ = x_
        self.interval = (x_ - x) / end

    def __call__(self, current: int) -> float:
        '''
        TODO: Annotation
        '''
        return max(self.x + self.interval * current, self.x_)
