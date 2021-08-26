#!/usr/bin/env python3
# encoding: utf-8

import os
import yaml

from typing import Dict, NoReturn
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


def load_config(filename: str) -> Dict:
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            x = yaml.safe_load(f.read())
        logger.info(
            colorize(f'load config from {filename} successfully', color='green'))
        return x or {}
    else:
        raise Exception('cannot find this config.')
