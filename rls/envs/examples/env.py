

from typing import Dict, List, NoReturn

import numpy as np

from rls.common.specs import Data, EnvAgentSpec, SensorSpec
from rls.envs.env_base import EnvBase


class ExampleEnv(EnvBase):

    def __init__(self):
        raise NotImplementedError

    def reset(self, **kwargs) -> Dict[str, Data]:
        raise NotImplementedError

    def step(self, actions: Dict[str, np.ndarray], **kwargs) -> Dict[str, Data]:
        raise NotImplementedError

    def close(self, **kwargs) -> NoReturn:
        raise NotImplementedError

    def render(self, **kwargs) -> NoReturn:
        raise NotImplementedError

    @property
    def n_copys(self) -> int:
        raise NotImplementedError

    @property
    def AgentSpecs(self) -> Dict[str, EnvAgentSpec]:
        raise NotImplementedError

    @property
    def StateSpec(self) -> SensorSpec:
        raise NotImplementedError

    @property
    def is_multi(self) -> bool:
        raise NotImplementedError

    @property
    def agent_ids(self) -> List[str]:
        raise NotImplementedError

    # TODO: implement
    def evaluate(self, model=None):
        raise NotImplementedError

    def run_trajectories(self, model=None):
        raise NotImplementedError
