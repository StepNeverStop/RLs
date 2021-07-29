
import numpy as np

from typing import (List,
                    NoReturn)

from rls.envs.env_base import EnvBase
from rls.common.specs import (EnvGroupArgs,
                              ModelObservations,
                              SingleModelInformation)
from rls.envs.unity.wrappers import (BasicUnityEnvironment,
                                     ScaleVisualWrapper)


class UnityEnv(EnvBase):

    def __init__(self,
                 obs_scale=False,
                 **kwargs):
        self.env = BasicUnityEnvironment(**kwargs)
        if obs_scale:
            self.env = ScaleVisualWrapper(env)

    def reset(self, reset_config={}, **kwargs) -> List[ModelObservations]:
        return self.env.reset(reset_config)

    def step(self, actions: List[np.ndarray], step_config={}, **kwargs) -> List[SingleModelInformation]:
        return self.env.step(actions, step_config)

    def close(self, **kwargs) -> NoReturn:
        return self.env.close()

    def random_action(self, **kwargs) -> List[np.ndarray]:
        return self.env.random_action()

    def render(self, **kwargs) -> NoReturn:
        pass

    @property
    def n_agents(self) -> int:
        return self.env.n_agents

    @property
    def n_copys(self) -> int:
        return int(self.env._n_copys)

    @property
    def GroupsSpec(self) -> List[EnvGroupArgs]:
        return self.env.GroupsSpec

    @property
    def is_multi(self) -> bool:
        return len(self.GroupsSpec) > 1
