import yaml
from mlagents.envs.sampler_class import SamplerManager
from mlagents.envs.exception import UnityEnvironmentException, SamplerException
from typing import Any, Callable, Dict, Optional


def create_sampler_manager(sampler_file_path, env_reset_params):
    '''
    resample_interval refer to episodes between last param and next one.
    '''
    sampler_config = None
    resample_interval = float("inf")
    if sampler_file_path is not None:
        sampler_config = load_config(sampler_file_path)
        if ("resampling-interval") in sampler_config:
            # Filter arguments that do not exist in the environment
            resample_interval = sampler_config.pop("resampling-interval")
            if (resample_interval <= 0) or (not isinstance(resample_interval, int)):
                raise SamplerException(
                    "Specified resampling-interval is not valid. Please provide"
                    " a positive integer value for resampling-interval"
                )
        else:
            raise SamplerException(
                "Resampling interval was not specified in the sampler file."
                " Please specify it with the 'resampling-interval' key in the sampler config file."
            )
    sampler_manager = SamplerManager(sampler_config)
    return sampler_manager, resample_interval


def load_config(trainer_config_path: str) -> Dict[str, Any]:
    try:
        with open(trainer_config_path) as data_file:
            trainer_config = yaml.safe_load(data_file)
            return trainer_config
    except IOError:
        raise UnityEnvironmentException(
            "Parameter file could not be found " "at {}.".format(trainer_config_path)
        )
    except UnicodeDecodeError:
        raise UnityEnvironmentException(
            "There was an error decoding "
            "Trainer Config from this path : {}".format(trainer_config_path)
        )
