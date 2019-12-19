import os
import yaml

def load_yaml(rel_filepath):
    '''Load YAML file.
    '''
    if os.path.exists(rel_filepath):
        with open(rel_filepath, 'r', encoding='utf-8') as f:
            x = yaml.safe_load(f.read())
        return x
    else:
        raise Exception('cannot find this config.')