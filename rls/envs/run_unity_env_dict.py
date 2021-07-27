#!/usr/bin/env python3
# encoding: utf-8

import os
import yaml

INPUT_FILE = '*.asset'
OUTPUT_FILE = 'rls/configs/unity/env_dict.yaml'


def get_env_dict(in_path, out_path):

    all_env_path = []
    if os.path.exists(in_path):
        with open(in_path, 'r') as f:
            for line in f:
                txt = line.strip()
                if txt[:5] == 'path:':
                    all_env_path.append(txt.split('Assets/')[-1].replace('.unity', ''))
    env_dict = {'Official' + path.split('/')[-1] if 'ML-Agents' in path else path.split('/')[-1]: path for path in all_env_path}
    with open(out_path, 'w') as f:
        f.write(yaml.dump(env_dict))
    pass


if __name__ == "__main__":
    get_env_dict(INPUT_FILE, OUTPUT_FILE)
    pass
