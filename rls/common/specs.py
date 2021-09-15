
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class SensorSpec:
    vector_dims: Optional[List[int]] = None
    visual_dims: Optional[List[Union[List[int], Tuple[int]]]] = None
    other_dims: int = 0

    @property
    def has_vector_observation(self):
        return self.vector_dims is not None and len(self.vector_dims) > 0

    @property
    def has_visual_observation(self):
        return self.visual_dims is not None and len(self.visual_dims) > 0

    @property
    def has_other_observation(self):
        return self.other_dims > 0


@dataclass
class EnvAgentSpec:
    obs_spec: SensorSpec
    a_dim: int
    is_continuous: bool
