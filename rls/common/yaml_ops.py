#!/usr/bin/env python3
# encoding: utf-8

import os
from typing import Dict, NoReturn

import yaml

from rls.utils.display import colorize
from rls.utils.logging_utils import get_logger

logger = get_logger(__name__)


def save_config(dicpath: str, config: Dict, filename: str) -> NoReturn:
    if not os.path.exists(dicpath):
        os.makedirs(dicpath)
    with open(os.path.join(dicpath, filename), 'w', encoding='utf-8') as fw:
        yaml.dump(config, fw)
    logger.info(
        colorize(f'save config to {dicpath} successfully', color='green'))


def load_config(filename: str, not_find_error=True) -> Dict:
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            x = yaml.safe_load(f.read())
        logger.info(
            colorize(f'load config from {filename} successfully', color='green'))
        return x or {}
    else:
        if not_find_error:
            raise Exception('cannot find this config.')
        else:
            logger.info(
                colorize(f'load config from {filename} failed, cannot find file.', color='red'))
