

import numpy as np

from typing import (List,
                    NoReturn)
from abc import ABC, abstractmethod

from rls.common.specs import (EnvGroupArgs,
                              ModelObservations,
                              SingleModelInformation)


class EnvBase(ABC):

    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self, **kwargs) -> List[ModelObservations]:
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: List[np.ndarray], **kwargs) -> List[SingleModelInformation]:
        raise NotImplementedError

    @abstractmethod
    def close(self, **kwargs) -> NoReturn:
        raise NotImplementedError

    @abstractmethod
    def random_action(self, **kwargs) -> List[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def render(self, **kwargs) -> NoReturn:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_agents(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def n_copys(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def GroupsSpec(self) -> List[EnvGroupArgs]:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_multi(self) -> bool:
        raise NotImplementedError

    # TODO: implement
    def evaluate(self, model=None):
        raise NotImplementedError

    def run_trajectories(self, model=None):
        raise NotImplementedError
