#!/usr/bin/env python3
# encoding: utf-8

from abc import abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, List, NoReturn, Optional, Union

import numpy as np
import torch as t

from rls.algorithms.base.policy import Policy
from rls.common.specs import Data, EnvAgentSpec, SensorSpec
from rls.utils.converter import to_tensor
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
        self.obs_specs = {id: agent_spec.obs_spec for id,
                          agent_spec in agent_specs.items()}
        self.is_continuouss = {
            id: agent_spec.is_continuous for id, agent_spec in agent_specs.items()}
        self.a_dims = {id: agent_spec.a_dim for id,
                       agent_spec in agent_specs.items()}

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
        self.agent_writers = {id: self._create_writer(
            self.log_dir + f'_{id}') for id in self.agent_ids}

    def _preprocess_obs(self, obs: Dict):
        for i, id in enumerate(self.agent_ids):
            other = None
            if self._obs_with_pre_action:
                if not self.is_continuouss[id]:
                    other = int2one_hot(self._pre_acts[id], self.a_dims[id])
                else:
                    other = self._pre_acts[id]
            if self._obs_with_agent_id:
                _id_onehot = int2one_hot(
                    np.full(self.n_copys, i), self.n_agents_percopy)
                if other is not None:
                    other = np.concatenate((
                        other,
                        _id_onehot
                    ), -1)
                else:
                    other = _id_onehot
            obs[id].update(other=other)
        return obs

    def __call__(self, obs):
        obs = self._preprocess_obs(obs)
        self._pre_acts, self._acts_info = self.select_action(obs)
        return self._pre_acts

    def select_action(self, obs):
        raise NotImplementedError

    def random_action(self):
        actions = {}
        self._acts_info = {}
        for id in self.agent_ids:
            if self.is_continuouss[id]:
                actions[id] = np.random.uniform(-1.0,
                                                1.0, (self.n_copys, self.a_dims[id]))
            else:
                actions[id] = np.random.randint(
                    0, self.a_dims[id], self.n_copys)
            self._acts_info[id] = Data(action=actions[id])
        self._pre_acts = actions
        return actions

    def episode_reset(self):
        self._pre_acts = {}
        for id in self.agent_ids:
            self._pre_acts[id] = np.zeros(
                (self.n_copys, self.a_dims[id])) if self.is_continuouss[id] else np.zeros(self.n_copys)
        self.cell_state, self.next_cell_state = {}, {}
        for id in self.agent_ids:
            self.cell_state[id] = to_tensor(self._initial_cell_state(
                batch=self.n_copys), device=self.device)
            self.next_cell_state[id] = to_tensor(self._initial_cell_state(
                batch=self.n_copys), device=self.device)

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
                                 obs_=env_rets[id].obs,
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
            self.cell_state[id] = self.next_cell_state[id]
            if self.cell_state[id] is not None:
                for k in self.cell_state[id].keys():
                    self.cell_state[id][k][idxs] = 0.

    def write_recorder_summaries(self, summaries):
        if 'model' in summaries.keys():
            super()._write_train_summaries(self._cur_episode,
                                           summaries=summaries.pop('model'), writer=self.writer)
        for id, summary in summaries.items():
            super()._write_train_summaries(self._cur_episode,
                                           summaries=summary, writer=self.agent_writers[id])

    # customed

    def _train(self, BATCH_DICT):
        raise NotImplementedError

    def _write_train_summaries(self,
                               cur_train_step: Union[int, t.Tensor],
                               summaries: Dict) -> NoReturn:
        '''
        write summaries showing in tensorboard.
        '''
        if 'model' in summaries.keys():
            super()._write_train_summaries(cur_train_step,
                                           summaries=summaries.pop('model'), writer=self.writer)
        for id, summary in summaries.items():
            super()._write_train_summaries(cur_train_step,
                                           summaries=summary, writer=self.agent_writers[id])
