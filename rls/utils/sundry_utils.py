#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

from typing import \
    List, \
    NoReturn

from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


def check_or_create(dicpath: str, name: str = '') -> NoReturn:
    """
    check dictionary whether existing, if not then create it.
    """
    if not os.path.exists(dicpath):
        os.makedirs(dicpath)
        logger.info(''.join([f'create {name} directionary :', dicpath]))


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
