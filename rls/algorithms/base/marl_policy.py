#!/usr/bin/env python3
# encoding: utf-8

from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, NoReturn, Optional, Union

import numpy as np
import torch as t

from rls.algorithms.base.policy import Policy
from rls.common.data import Data
from rls.common.specs import EnvAgentSpec, SensorSpec
from rls.utils.converter import to_tensor
from rls.utils.loggers import Log_REGISTER
from rls.utils.np_utils import int2one_hot


class MarlPolicy(Policy):

    def __init__(self,
                 agent_specs: Dict[str, EnvAgentSpec],
                 state_spec: SensorSpec,

                 share_params=True,
                 obs_with_agent_id=True,
                 **kwargs):
        self.agent_specs = agent_specs
        self.n_agents_percopy = len(agent_specs)
        self.agent_ids = list(self.agent_specs.keys())
        self.obs_specs = {id: agent_spec.obs_spec for id, agent_spec in agent_specs.items()}
        self.is_continuouss = {id: agent_spec.is_continuous for id, agent_spec in agent_specs.items()}
        self.a_dims = {id: agent_spec.a_dim for id, agent_spec in agent_specs.items()}

        self.state_spec = state_spec
        self.share_params = share_params
        self._obs_with_agent_id = obs_with_agent_id

        super().__init__(**kwargs)

        self._has_global_state = self.state_spec.has_vector_observation or self.state_spec.has_visual_observation

        if self._obs_with_pre_action:
            for id in self.agent_ids:
                self.obs_specs[id].other_dims += self.a_dims[id]
        if self._obs_with_agent_id:
            for id in self.agent_ids:
                self.obs_specs[id].other_dims += self.n_agents_percopy

        self.model_ids = self.agent_ids.copy()

        if self.share_params:
            for i in range(self.n_agents_percopy):
                for id in self.agent_ids[:i]:
                    if self.agent_specs[self.agent_ids[i]] == self.agent_specs[id]:
                        self.model_ids[i] = id
                        break

    def _build_loggers(self):
        return [
            Log_REGISTER[logger_type](
                log_dir=self.log_dir,
                ids=['model'] + self.agent_ids,
                training_name=self._training_name,  # wandb
            )
            for logger_type in self._logger_types
        ]

    def _preprocess_obs(self, obs: Dict):
        for i, id in enumerate(self.agent_ids):
            other = None
            if self._obs_with_pre_action:
                if not self.is_continuouss[id]:
                    other = int2one_hot(self._pre_acts[id], self.a_dims[id])
                else:
                    other = self._pre_acts[id]
            if self._obs_with_agent_id:
                _id_onehot = int2one_hot(np.full(self.n_copys, i), self.n_agents_percopy)
                if other is not None:
                    other = np.concatenate((
                        other,
                        _id_onehot
                    ), -1)
                else:
                    other = _id_onehot
            if self._obs_with_pre_action or self._obs_with_agent_id:
                obs[id].update(other=other)
        return obs

    def __call__(self, obs):
        obs = self._preprocess_obs(deepcopy(obs))
        self._pre_acts, self._acts_info = self.select_action(obs)
        return self._pre_acts

    def select_action(self, obs):
        raise NotImplementedError

    def random_action(self):
        actions = {}
        self._acts_info = {}
        for id in self.agent_ids:
            if self.is_continuouss[id]:
                actions[id] = np.random.uniform(-1.0, 1.0, (self.n_copys, self.a_dims[id]))
            else:
                actions[id] = np.random.randint(0, self.a_dims[id], self.n_copys)
            self._acts_info[id] = Data(action=actions[id])
        self._pre_acts = actions
        return actions

    def episode_reset(self):
        self._pre_acts = {}
        for id in self.agent_ids:
            self._pre_acts[id] = np.zeros((self.n_copys, self.a_dims[id])) if self.is_continuouss[id] else np.zeros(self.n_copys)
        self.rnncs, self.rnncs_ = {}, {}
        for id in self.agent_ids:
            self.rnncs[id] = to_tensor(self._initial_rnncs(batch=self.n_copys), device=self.device)
            self.rnncs_[id] = to_tensor(self._initial_rnncs(batch=self.n_copys), device=self.device)

    def episode_step(self,
                     obs,
                     env_rets: Dict[str, Data]):
        super().episode_step()
        if self._store:
            expss = {}
            for id in self.agent_ids:
                expss[id] = Data(obs=obs[id],
                                 # [B, ] => [B, 1]
                                 reward=env_rets[id].reward[:, np.newaxis],
                                 obs_=env_rets[id].obs_fs,
                                 done=env_rets[id].done[:, np.newaxis])
                expss[id].update(self._acts_info[id])
            expss['global'] = Data(begin_mask=obs['global'].begin_mask)
            if self._has_global_state:
                expss['global'].update(obs=obs['global'].obs,
                                       obs_=env_rets['global'].obs)
            self._buffer.add(expss)

        for id in self.agent_ids:
            idxs = np.where(env_rets[id].done)[0]
            self._pre_acts[id][idxs] = 0.
            self.rnncs[id] = self.rnncs_[id]
            if self.rnncs[id] is not None:
                for k in self.rnncs[id].keys():
                    self.rnncs[id][k][idxs] = 0.

    # customed

    def _train(self, BATCH_DICT):
        raise NotImplementedError
