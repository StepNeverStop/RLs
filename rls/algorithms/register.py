#!/usr/bin/env python3
# encoding: utf-8

# algorithms based on PyTorch

import importlib
from typing import Callable, Dict, Tuple

from rls.common.yaml_ops import load_config
from rls.utils.display import colorize
from rls.utils.logging_utils import get_logger

logger = get_logger(__name__)


class AlgoRegistry(object):

    def __init__(self):
        self.algo_specs = {}

    def register(self, name, **attrs):
        if name in self.algo_specs.keys():
            raise Exception(f'Cannot re-register algorithms: {name}')
        self.algo_specs[name] = dict(attrs)

    def get_model_info(self, name):
        if name in self.algo_specs.keys():
            return self.algo_specs[name]
        raise Exception(f'Cannot find algorithm: {name}')


registry = AlgoRegistry()


def register(name, **attrs):
    registry.register(name, **attrs)


def get_model_info(name: str) -> Tuple[Callable, Dict, str, str]:
    '''
    Args:
        name: name of algorithms
    Return:
        class of the algorithm model
        whether algorithm is sarl or marl
    '''
    algo_info = registry.get_model_info(name)
    logger.info(colorize(algo_info.get('logo', ''), color='green'))
    model_class = getattr(importlib.import_module(
        f"rls.algorithms.{algo_info['path']}"), algo_info['class_name'])
    return model_class, algo_info['is_multi']
