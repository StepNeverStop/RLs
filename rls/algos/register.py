#!/usr/bin/env python3
# encoding: utf-8

import importlib
import tensorflow as tf
assert tf.__version__[0] == '2'

# algorithms based on TF 2.x

from typing import (Tuple,
                    Callable,
                    Dict)

from rls.common.yaml_ops import load_yaml
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
        algo_class of the algorithm model named `name`.
        defaulf config of specified algorithm.
        policy_type of policy, `on-policy` or `off-policy`
    '''
    algo_info = registry.get_model_info(name)
    class_name = algo_info['algo_class']
    policy_mode = algo_info['policy_mode']
    policy_type = algo_info['policy_type']
    LOGO = algo_info.get('logo', '')
    logger.info(colorize(LOGO, color='green'))

    model = getattr(
        importlib.import_module(f'rls.algos.{policy_type}.{name}'),
        class_name)

    algo_config = {}
    algo_config.update(
        load_yaml(f'rls/algos/config.yaml')['general']
    )
    algo_config.update(
        load_yaml(f'rls/algos/config.yaml')[policy_mode.replace('-', '_')]
    )
    algo_config.update(
        load_yaml(f'rls/algos/config.yaml')[name]
    )
    return model, algo_config, policy_mode, policy_type
