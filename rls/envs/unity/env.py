from typing import Dict, List, NoReturn

import numpy as np

from rls.common.data import Data
from rls.common.specs import EnvAgentSpec, SensorSpec
from rls.envs.env_base import EnvBase
from rls.envs.unity.wrappers import BasicUnityEnvironment, ScaleVisualWrapper


class UnityEnv(EnvBase):

    def __init__(self,
                 obs_scale=False,
                 **kwargs):
        super().__init__()
        self.env = BasicUnityEnvironment(**kwargs)
        if obs_scale:
            self.env = ScaleVisualWrapper(self.env)

    def reset(self, reset_config={}, **kwargs) -> Dict[str, Data]:
        return self.env.reset(reset_config)

    def step(self, actions: Dict[str, np.ndarray], step_config={}, **kwargs) -> Dict[str, Data]:
        return self.env.step(actions, step_config)

    def close(self, **kwargs) -> NoReturn:
        return self.env.close()

    def render(self, **kwargs) -> NoReturn:
        pass

    @property
    def n_copies(self) -> int:
        return int(self.env._n_copies)

    @property
    def AgentSpecs(self) -> Dict[str, EnvAgentSpec]:
        return self.env.AgentSpecs

    @property
    def StateSpec(self) -> SensorSpec:
        return self.env.StateSpec

    @property
    def is_multi(self) -> bool:  # TODO: optimize
        return len(self.agent_ids) > 1

    @property
    def agent_ids(self) -> List[str]:
        return self.env.agent_ids
