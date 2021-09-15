#!/usr/bin/env python3
# encoding: utf-8

from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, Union

import numpy as np
import torch as t

from rls.algorithms.base.policy import Policy
from rls.common.data import Data
from rls.common.specs import EnvAgentSpec
from rls.nn.modules import CuriosityModel
from rls.utils.converter import to_tensor
from rls.utils.loggers import Log_REGISTER
from rls.utils.np_utils import int2one_hot
from rls.utils.vector_runing_average import (DefaultRunningAverage,
                                             SimpleRunningAverage)


class SarlPolicy(Policy):
    def __init__(self,
                 agent_spec: EnvAgentSpec,
                 agent_id: str,

                 use_curiosity=False,
                 curiosity_reward_eta=0.01,
                 curiosity_lr=1.0e-3,
                 curiosity_beta=0.2,
                 **kwargs):
        super().__init__(**kwargs)

        self.agent_spec = agent_spec
        self._agent_id = agent_id
        self.obs_spec = agent_spec.obs_spec
        self.is_continuous = agent_spec.is_continuous
        self.a_dim = agent_spec.a_dim
        if self._obs_with_pre_action:
            self.obs_spec.other_dims += self.a_dim

        # self._normalize_vector_obs = normalize_vector_obs
        # self._running_average = SimpleRunningAverage(dim=None) if self._normalize_vector_obs else DefaultRunningAverage()

        self.use_curiosity = use_curiosity
        if self.use_curiosity:
            self.curiosity_model = CuriosityModel(self.obs_spec,
                                                  self._rep_net_params,
                                                  self.is_continuous,
                                                  self.a_dim,
                                                  eta=curiosity_reward_eta,
                                                  lr=curiosity_lr,
                                                  beta=curiosity_beta).to(self.device)
            self._trainer_modules.update(curiosity_model=self.curiosity_model)

    # def normalize_vector_obs(self, x: np.ndarray) -> np.ndarray:
    #     return self._running_average.normalize(x)

    def _preprocess_obs(self, obs):
        if self._obs_with_pre_action:
            if not self.is_continuous:
                _pre_act = int2one_hot(self._pre_act, self.a_dim)
            else:
                _pre_act = self._pre_act
            obs.update(other=_pre_act)
        return obs

    def __call__(self, obs):
        obs = self._preprocess_obs(deepcopy(obs))
        self._pre_act, self._acts_info = self.select_action(obs)
        return self._pre_act

    def select_action(self, obs):
        raise NotImplementedError

    def random_action(self):
        if self.is_continuous:
            actions = np.random.uniform(-1.0, 1.0, (self.n_copys, self.a_dim))
        else:
            actions = np.random.randint(0, self.a_dim, self.n_copys)
        self._pre_act = actions
        self._acts_info = Data(action=actions)
        return actions

    def episode_reset(self):
        '''reset model for each new episode.'''
        self._pre_act = np.zeros(
            (self.n_copys, self.a_dim)) if self.is_continuous else np.zeros(self.n_copys)
        self.rnncs = to_tensor(self._initial_rnncs(
            batch=self.n_copys), device=self.device)
        self.rnncs_ = to_tensor(self._initial_rnncs(
            batch=self.n_copys), device=self.device)

    def episode_step(self,
                     obs: Data,
                     env_rets: Data,
                     begin_mask: np.ndarray):
        super().episode_step()
        if self._store:
            exps = Data(obs=obs,
                        # [B, ] => [B, 1]
                        reward=env_rets.reward[:, np.newaxis],
                        obs_=env_rets.obs_fs,
                        done=env_rets.done[:, np.newaxis],
                        begin_mask=begin_mask)
            exps.update(self._acts_info)
            self._buffer.add({self._agent_id: exps})

        idxs = np.where(env_rets.done)[0]
        self._pre_act[idxs] = 0.
        self.rnncs = self.rnncs_
        if self.rnncs is not None:
            for k in self.rnncs.keys():
                self.rnncs[k][idxs] = 0.

    def learn(self, BATCH: Data):
        raise NotImplementedError

    # customed

    def _train(self, BATCH):
        raise NotImplementedError

    def _build_loggers(self):
        return [
            Log_REGISTER[logger_type](
                log_dir=self.log_dir,
                training_name=self._training_name,  # wandb
            )
            for logger_type in self._logger_types
        ]
