

import numpy as np

from typing import (List,
                    NoReturn)

from rls.envs.env_base import EnvBase
from rls.common.specs import (EnvGroupArgs,
                              ModelObservations,
                              SingleModelInformation)


class ExampleEnv(EnvBase):

    def __init__(self):
        raise NotImplementedError

    def reset(self, **kwargs) -> List[ModelObservations]:
        raise NotImplementedError

    def step(self, actions: List[np.ndarray], **kwargs) -> List[SingleModelInformation]:
        raise NotImplementedError

    def close(self, **kwargs) -> NoReturn:
        raise NotImplementedError

    def random_action(self, **kwargs) -> List[np.ndarray]:
        raise NotImplementedError

    def render(self, **kwargs) -> NoReturn:
        raise NotImplementedError

    @property
    def n_agents(self) -> int:
        raise NotImplementedError

    @property
    def n_copys(self) -> int:
        raise NotImplementedError

    @property
    def GroupsSpec(self) -> List[EnvGroupArgs]:
        raise NotImplementedError

    @property
    def is_multi(self) -> bool:
        raise NotImplementedError

    # TODO: implement
    def evaluate(self, model=None):
        raise NotImplementedError

    def run_trajectories(self, model=None):
        raise NotImplementedError
