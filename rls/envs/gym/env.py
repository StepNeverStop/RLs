from copy import deepcopy
from typing import Dict, List, NoReturn

import numpy as np
from gym.spaces import Box, Discrete, Tuple

from rls.common.data import Data
from rls.common.specs import EnvAgentSpec, SensorSpec
from rls.envs.env_base import EnvBase
from rls.envs.gym.make_env import make_env
from rls.envs.wrappers import MPIEnv, VECEnv
from rls.utils.display import colorize
from rls.utils.logging_utils import get_logger

logger = get_logger(__name__)

try:
    import gym_minigrid
except ImportError:
    logger.warning(
        colorize("import gym_minigrid failed, using 'pip3 install gym-minigrid' install it.", color='yellow'))
    pass

try:
    # if wanna render, added 'renders=True' or(depends on env) 'render=True' in gym.make() function manually.
    import pybullet_envs
except ImportError:
    logger.warning(colorize("import pybullet_envs failed, using 'pip3 install PyBullet' install it.", color='yellow'))
    pass

try:
    import gym_donkeycar
except ImportError:
    logger.warning(
        colorize("import gym_minigrid failed, using 'pip install gym_donkeycar' install it.", color='yellow'))
    pass

try:
    import highway_env
except ImportError:
    logger.warning(colorize(
        "import highway_env failed, using 'pip install --user git+https://github.com/eleurent/highway-env' install it.",
        color='yellow'))
    pass


class GymEnv(EnvBase):

    def __init__(self,
                 env_copies=1,
                 multiprocessing=True,
                 seed=42,
                 inference=False,
                 **kwargs):
        """
        Input:
            env_copies: environment number
        """
        super().__init__()
        self._n_copies = env_copies  # environments number
        self._initialize(env=make_env(**kwargs))
        _env_wrapper = MPIEnv if multiprocessing else VECEnv
        self._envs = _env_wrapper(n=self._n_copies, env_fn=make_env, config=kwargs)

        params = []
        for i in range(self._n_copies):
            params.append(dict(args=(seed + i,)))
        self._envs.run('seed', params)

    def reset(self, **kwargs) -> Dict[str, Data]:
        obs = self._envs.run('reset')
        obs = np.stack(obs, 0)
        if self._use_visual:
            ret = Data(visual={'visual_0': obs})
        else:
            ret = Data(vector={'vector_0': obs})
        return {'single': ret,
                'global': Data(begin_mask=np.full((self._n_copies, 1), True))}

    def step(self, actions: Dict[str, np.ndarray], **kwargs) -> Dict[str, Data]:
        # choose the first agents' actions
        actions = deepcopy(actions['single'])
        params = []
        for i in range(self._n_copies):
            params.append(dict(args=(actions[i],)))
        rets = self._envs.run('step', params)
        obs_fs, reward, done, info = zip(*rets)
        obs_fs = np.stack(obs_fs, 0)
        reward = np.stack(reward, 0)
        done = np.stack(done, 0)
        # TODO: info

        obs_fa = deepcopy(obs_fs)  # obs for next action choosing.

        idxs = np.where(done)[0]
        if len(idxs) > 0:
            reset_obs = self._envs.run('reset', idxs=idxs)
            obs_fa[idxs] = np.stack(reset_obs, 0)

        if self._use_visual:
            obs_fs = Data(visual={'visual_0': obs_fs})
            obs_fa = Data(visual={'visual_0': obs_fa})
        else:
            obs_fs = Data(vector={'vector_0': obs_fs})
            obs_fa = Data(vector={'vector_0': obs_fa})
        return {'single': Data(obs_fs=obs_fs,
                               obs_fa=obs_fa,
                               reward=reward,
                               done=done,
                               info=info),
                'global': Data(begin_mask=done[:, np.newaxis])}

    def close(self, **kwargs) -> NoReturn:
        """
        close all environments.
        """
        self._envs.run('close')

    def render(self, **kwargs) -> NoReturn:
        """
        render game windows.
        """
        self._envs.run('render', idxs=0)

    @property
    def n_copies(self) -> int:
        return int(self._n_copies)

    @property
    def AgentSpecs(self) -> Dict[str, EnvAgentSpec]:
        return {'single': EnvAgentSpec(
            obs_spec=SensorSpec(vector_dims=self._vector_dims,
                                visual_dims=self._visual_dims),
            a_dim=self.a_dim,
            is_continuous=self._is_continuous
        )}

    @property
    def StateSpec(self) -> SensorSpec:
        return SensorSpec()

    @property
    def is_multi(self) -> bool:
        return False

    @property
    def agent_ids(self) -> List[str]:
        return ['single']

    # --- custom

    def _initialize(self, env):
        assert isinstance(env.observation_space, (Box, Discrete)) and isinstance(env.action_space, (
            Box, Discrete)), 'action_space and observation_space must be one of available_type'
        # process observation
        ObsSpace = env.observation_space

        self._use_visual = False

        if isinstance(ObsSpace, Box):
            if len(ObsSpace.shape) == 1:
                self._vector_dims = list(ObsSpace.shape)
                self._visual_dims = []
            elif len(ObsSpace.shape) == 3:
                self._vector_dims = []
                self._visual_dims = [list(ObsSpace.shape)]
                self._use_visual = True
            else:
                raise ValueError
        else:
            self._vector_dims = [int(ObsSpace.n)]
            self._visual_dims = []

        # process action
        ActSpace = env.action_space
        if isinstance(ActSpace, Box):
            assert len(
                ActSpace.shape) == 1, 'if action space is continuous, the shape length of action must equal to 1'
            self._is_continuous = True
            self.a_dim = ActSpace.shape[0]
        elif isinstance(ActSpace, Tuple):
            assert all([isinstance(i, Discrete) for i in
                        ActSpace]) == True, 'if action space is Tuple, each item in it must have type Discrete'
            self._is_continuous = False
            self.a_dim = int(np.asarray([i.n for i in ActSpace]).prod())
        else:
            self._is_continuous = False
            self.a_dim = ActSpace.n
        env.close()
