import yaml
from mlagents.trainers.sampler_class import SamplerManager
from mlagents.trainers.exception import SamplerException
from typing import Any, Dict, TextIO


def create_sampler_manager(sampler_path, run_seed=None):
    sampler_config = load_config(sampler_path)
    resample_interval = float('inf')
    if sampler_config is not None:
        if "resampling-interval" in sampler_config:
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
    sampler_manager = SamplerManager(sampler_config, run_seed)
    return sampler_manager, resample_interval


def load_config(config_path: str) -> Dict[str, Any]:
    if config_path is None:
        return None
    try:
        with open(config_path) as data_file:
            return _load_config(data_file)
    except IOError:
        abs_path = os.path.abspath(config_path)
        raise TrainerConfigError(f"Config file could not be found at {abs_path}.")
    except UnicodeDecodeError:
        raise TrainerConfigError(
            f"There was an error decoding Config file from {config_path}. "
            f"Make sure your file is save using UTF-8"
        )


def _load_config(fp: TextIO) -> Dict[str, Any]:
    """
    Load the yaml config from the file-like object.
    """
    try:
        return yaml.safe_load(fp)
    except yaml.parser.ParserError as e:
        raise TrainerConfigError(
            "Error parsing yaml file. Please check for formatting errors. "
            "A tool such as http://www.yamllint.com/ can be helpful with this."
        ) from e
