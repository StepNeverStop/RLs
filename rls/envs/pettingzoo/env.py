import importlib
from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, NoReturn

import numpy as np
import pettingzoo
from gym.spaces import Box, Discrete, Tuple

from rls.common.specs import Data, EnvAgentSpec, SensorSpec
from rls.envs.env_base import EnvBase
from rls.envs.pettingzoo.wrappers import BasicWrapper


class PettingZooEnv(EnvBase):

    def __init__(self,
                 env_copys=1,
                 seed=42,
                 multiprocessing=True,
                 env_name='mpe.simple_v2',
                 env_config={},
                 **kwargs):

        self._n_copys = env_copys   # environments number

        # NOTE: env_name should be formatted like `mpe.simple_v2`
        env_module = importlib.import_module(f"pettingzoo.{env_name}")
        env_module = env_module.parallel_env
        # TODO:
        def make_env(
            idx, **config): return BasicWrapper(env_module(**env_config))

        self._initialize(env=env_module(**env_config))
        self._envs = [env_module(**env_config) for _ in range(self._n_copys)]
        [env.seed(seed+i) for i, env in enumerate(self._envs)]

    def reset(self, **kwargs) -> Dict[str, Data]:
        obss = [env.reset() for env in self._envs]
        _obs = defaultdict(list)
        for k in self._agents:
            for obs in obss:
                _obs[k].append(obs[k])
            _obs[k] = np.asarray(_obs[k])   # [B, *]

        rets = {}
        for k in self._agents:
            if self._is_obs_visual[k]:
                rets[k] = Data(visual={'visual_0': _obs[k]})
            else:
                rets[k] = Data(vector={'vector_0': _obs[k]})
        rets['global'] = Data(begin_mask=np.full((self._n_copys, 1), True))

        if self._has_global_state:
            state = np.asarray([env.state() for env in self._envs])  # [B, *]
            if self._is_state_visual:
                _state = Data(visual={'visual_0': state})
            else:
                _state = Data(vector={'vector_0': state})
            rets['global'].update(obs=_state)

        return rets

    def step(self, actions: Dict[str, np.ndarray], **kwargs) -> Dict[str, Data]:
        actions = deepcopy(actions)   # [N, B, *]

        obss, rewards, dones, infos = defaultdict(list), defaultdict(
            list), defaultdict(list), defaultdict(list)
        begin_mask = np.full((self._n_copys, 1), False)

        for i, env in enumerate(self._envs):
            action = {}
            for k in self._agents:
                action[k] = actions[k][i]
            obs, reward, done, info = env.step(action)

            if any(list(done.values())):
                obs = env.reset()
                begin_mask[i] = True

            for k in self._agents:
                obss[k].append(obs[k])
                rewards[k].append(reward[k])
                dones[k].append(done[k])
                infos[k].append(infos[k])

        for k in self._agents:
            obss[k] = np.asarray(obss[k])   # [B, *]
            rewards[k] = np.asarray(rewards[k])  # [B, *]
            dones[k] = np.asarray(dones[k])  # [B, *]

        rets = {}
        for k in self._agents:
            if self._is_obs_visual[k]:
                _obs = Data(visual={'visual_0': obss[k]})
            else:
                _obs = Data(vector={'vector_0': obss[k]})
            rets[k] = Data(obs=_obs,
                           reward=rewards[k],
                           done=dones[k],
                           info=infos[k])
        rets['global'] = Data(begin_mask=begin_mask)

        if self._has_global_state:
            state = np.asarray([env.state() for env in self._envs])
            if self._is_state_visual:
                _state = Data(visual={'visual_0': state})
            else:
                _state = Data(vector={'vector_0': state})
            rets['global'].update(obs=_state)
        return rets

    def close(self, **kwargs) -> NoReturn:
        [env.close() for env in self._envs]

    def render(self, **kwargs) -> NoReturn:
        self._envs[0].render()

    @property
    def n_copys(self) -> int:
        return self._n_copys

    @property
    def AgentSpecs(self) -> Dict[str, EnvAgentSpec]:
        return {k: EnvAgentSpec(
            obs_spec=SensorSpec(vector_dims=self._vector_dims[k],
                                visual_dims=self._visual_dims[k]),
            a_dim=self.a_dim[k],
            is_continuous=self._is_continuous[k]
        ) for k in self._agents}

    @property
    def StateSpec(self) -> SensorSpec:
        return SensorSpec(vector_dims=self._state_vector_dims,
                          visual_dims=self._state_visual_dims)

    @property
    def is_multi(self) -> bool:
        return len(self._agents) > 1

    @property
    def agent_ids(self) -> List[str]:
        return self._agents

    # --- custome

    def _initialize(self, env):
        self._is_obs_visual = {}
        self._vector_dims = defaultdict(list)
        self._visual_dims = defaultdict(list)

        # process state
        self._is_state_visual = False
        self._state_vector_dims = []
        self._state_visual_dims = []
        try:
            StateSpec = env.state_space
            if isinstance(StateSpec, Box):
                if len(StateSpec.shape) == 1:
                    self._state_vector_dims = list(StateSpec.shape)
                elif len(StateSpec.shape) == 3:
                    self._state_visual_dims = [list(StateSpec.shape)]
                    self._is_state_visual = True
            else:
                self._state_vector_dims = [int(StateSpec.n)]
            self._has_global_state = True
        except AttributeError:
            self._has_global_state = False
            pass

        self.a_dim = defaultdict(int)
        self._is_continuous = {}

        self._agents = env.possible_agents
        for k in env.possible_agents:
            # process observation
            ObsSpace = env.observation_spaces[k]
            self._is_obs_visual[k] = False
            if isinstance(ObsSpace, Box):
                if len(ObsSpace.shape) == 1:
                    self._vector_dims[k] = list(ObsSpace.shape)
                elif len(ObsSpace.shape) == 3:
                    self._visual_dims[k] = [list(ObsSpace.shape)]
                    self._is_obs_visual[k] = True
            else:
                self._vector_dims[k] = [int(ObsSpace.n)]

            # process action
            ActSpace = env.action_spaces[k]
            if isinstance(ActSpace, Box):
                assert len(
                    ActSpace.shape) == 1, 'if action space is continuous, the shape length of action must equal to 1'
                self._is_continuous[k] = True
                self.a_dim[k] = ActSpace.shape[0]
            elif isinstance(ActSpace, Tuple):
                assert all([isinstance(i, Discrete) for i in ActSpace]
                           ) == True, 'if action space is Tuple, each item in it must have type Discrete'
                self._is_continuous[k] = False
                self.a_dim[k] = int(np.asarray([i.n for i in ActSpace]).prod())
            else:
                self._is_continuous[k] = False
                self.a_dim[k] = ActSpace.n
        env.close()
