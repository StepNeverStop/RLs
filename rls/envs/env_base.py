

from abc import ABC, abstractmethod
from typing import Dict, List, NoReturn

import numpy as np

from rls.common.specs import Data, EnvAgentSpec, SensorSpec


class EnvBase(ABC):

    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self, **kwargs) -> Dict[str, Data]:
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: Dict[str, np.ndarray], **kwargs) -> Dict[str, Data]:
        raise NotImplementedError

    @abstractmethod
    def close(self, **kwargs) -> NoReturn:
        raise NotImplementedError

    @abstractmethod
    def render(self, **kwargs) -> NoReturn:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_copys(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def AgentSpecs(self) -> Dict[str, EnvAgentSpec]:
        raise NotImplementedError

    @property
    @abstractmethod
    def StateSpec(self) -> SensorSpec:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_multi(self) -> bool:
        raise NotImplementedError

    @property
    @abstractmethod
    def agent_ids(self) -> List[str]:
        raise NotImplementedError

    # TODO: implement
    def evaluate(self, model=None):
        raise NotImplementedError

    def run_trajectories(self, model=None):
        raise NotImplementedError
