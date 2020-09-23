#!/usr/bin/env python3
# encoding: utf-8

import os
import yaml

from typing import Dict, NoReturn


def load_yaml(rel_filepath: str, msg: str = '') -> Dict:
    '''
    Load YAML file.
    '''
    if os.path.exists(rel_filepath):
        with open(rel_filepath, 'r', encoding='utf-8') as f:
            x = yaml.safe_load(f.read())
        if msg != '':
            print(msg)
        return x
    else:
        raise Exception('cannot find this config.')


def save_config(dicpath: str, config: Dict) -> NoReturn:
    if not os.path.exists(dicpath):
        os.makedirs(dicpath)
    with open(os.path.join(dicpath, 'config.yaml'), 'w', encoding='utf-8') as fw:
        yaml.dump(config, fw)
    print(f'save config to {dicpath}')


def load_config(filename: str) -> Dict:
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            x = yaml.safe_load(f.read())
        print(f'load config from {filename}')
        return x
    else:
        raise Exception('cannot find this config.')
