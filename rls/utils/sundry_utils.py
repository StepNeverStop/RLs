#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import logging

from typing import \
    List, \
    NoReturn

from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)



def create_logger(
    name: str,
    console_level: int = logging.INFO,
    console_format: str = '%(levelname)s : %(message)s',
    logger2file: bool = False,
    file_name: str = './log.txt',
    file_level: int = logging.WARNING,
    file_format: str = '%(lineno)d - %(asctime)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s',
) -> logging.Logger:
    logger = logging.Logger(name)
    logger.setLevel(level=console_level)
    stdout_handle = logging.StreamHandler(stream=sys.stdout)
    stdout_handle.setFormatter(logging.Formatter(console_format if console_level > 20 else '%(message)s'))
    logger.addHandler(stdout_handle)
    if logger2file:
        logfile_handle = logging.FileHandler(file_name)
        logfile_handle.setLevel(file_level)
        logfile_handle.setFormatter(logging.Formatter(file_format))
        logger.addHandler(logfile_handle)
    return logger

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
