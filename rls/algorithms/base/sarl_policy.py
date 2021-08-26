#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from abc import abstractmethod
from collections import defaultdict
from typing import (Union,
                    List,
                    Callable,
                    Tuple,
                    Any,
                    Dict,
                    NoReturn,
                    Optional)

from rls.algorithms.base.policy import Policy
from rls.utils.vector_runing_average import (DefaultRunningAverage,
                                             SimpleRunningAverage)
from rls.common.specs import EnvAgentSpec
from rls.nn.modules import CuriosityModel
from rls.common.specs import Data
from rls.utils.np_utils import int2one_hot


class SarlPolicy(Policy):
    def __init__(self,
                 agent_spec: EnvAgentSpec,

                 use_curiosity=False,
                 curiosity_reward_eta=0.01,
                 curiosity_lr=1.0e-3,
                 curiosity_beta=0.2,
                 **kwargs):
        super().__init__(**kwargs)

        self.agent_spec = agent_spec
        self.obs_spec = agent_spec.obs_spec
        self.is_continuous = agent_spec.is_continuous
        self.a_dim = agent_spec.a_dim
        if self.obs_with_pre_action:
            self.obs_spec.other_dims += self.a_dim

        # self._normalize_vector_obs = normalize_vector_obs
        # self._running_average = SimpleRunningAverage(dim=self.obs_spec.total_vector_dim) if self._normalize_vector_obs else DefaultRunningAverage()

        self.use_curiosity = use_curiosity
        if self.use_curiosity:
            self.curiosity_model = CuriosityModel(self.obs_spec,
                                                  self.rep_net_params,
                                                  self.is_continuous,
                                                  self.a_dim,
                                                  eta=curiosity_reward_eta,
                                                  lr=curiosity_lr,
                                                  beta=curiosity_beta).to(self.device)
            self._trainer_modules.update(curiosity_model=self.curiosity_model)

    # def normalize_vector_obs(self, x: np.ndarray) -> np.ndarray:
    #     return self._running_average.normalize(x)

    def _preprocess_obs(self, obs):
        if self.obs_with_pre_action:
            if not self.is_continuous:
                _pre_act = int2one_hot(self._pre_act, self.a_dim)
            else:
                _pre_act = self._pre_act
            obs.update(other=_pre_act)
        return obs

    def __call__(self, obs):
        obs = self._preprocess_obs(obs)
        self._pre_act, acts = self.select_action(obs)
        return acts

    def select_action(self, obs):
        raise NotImplementedError

    def random_action(self):
        if self.is_continuous:
            return Data(action=np.random.uniform(-1.0, 1.0, (self.n_copys, self.a_dim)))
        else:
            return Data(action=np.random.randint(0, self.a_dim, self.n_copys))

    def episode_reset(self):
        '''reset model for each new episode.'''
        self._pre_act = np.zeros(
            (self.n_copys, self.a_dim)) if self.is_continuous else np.zeros(self.n_copys)
        self.cell_state = self._initial_cell_state(
            batch=self.n_copys, dtype='tensor')
        self.next_cell_state = self._initial_cell_state(
            batch=self.n_copys, dtype='tensor')

    def episode_step(self, done):
        super().episode_step()
        idxs = np.where(done)[0]
        self._pre_act[idxs] = 0.
        if self.next_cell_state is not None:
            for k in self.next_cell_state.keys():
                self.next_cell_state[k][idxs] = 0.
        self.cell_state = self.next_cell_state

    def learn(self, BATCH: Data):
        raise NotImplementedError

    def write_recorder_summaries(self, summaries):
        self._write_train_summaries(self.cur_episode, summaries, self.writer)

    # customed

    def _train(self, BATCH):
        raise NotImplementedError
