#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from torch import distributions as td
from dataclasses import dataclass

from rls.algorithms.base.on_policy import On_Policy
from rls.common.specs import (Data,
                              ModelObservations)
from rls.nn.models import (ActorMuLogstd,
                           ActorDct)
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


@dataclass(eq=False)
class A2C_Train_BatchExperiences(Data):
    obs: ModelObservations
    action: np.ndarray
    discounted_reward: np.ndarray


class PG(On_Policy):
    def __init__(self,
                 envspec,

                 lr=5.0e-4,
                 epoch=5,
                 network_settings={
                     'actor_continuous': {
                         'hidden_units': [32, 32],
                         'condition_sigma': False,
                         'log_std_bound': [-20, 2]
                     },
                     'actor_discrete': [32, 32]
                 },
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.epoch = epoch
        if self.is_continuous:
            self.net = ActorMuLogstd(self.rep_net.h_dim,
                                     output_shape=self.a_dim,
                                     network_settings=network_settings['actor_continuous']).to(self.device)
        else:
            self.net = ActorDct(self.rep_net.h_dim,
                                output_shape=self.a_dim,
                                network_settings=network_settings['actor_discrete']).to(self.device)
        self.oplr = OPLR([self.net, self.rep_net], lr)

        self.initialize_data_buffer(sample_data_type=PG_Train_BatchExperiences)

        self._worker_modules.update(rep_net=self.rep_net,
                                    model=self.net)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(oplr=self.oplr)

    def __call__(self, obs, evaluation=False):
        actions, self.next_cell_state = self.call(obs, cell_state=self.cell_state)
        return actions

    @iTensor_oNumpy
    def call(self, obs, cell_state):
        feat, cell_state = self.rep_net(obs, cell_state=cell_state)
        output = self.net(feat)
        if self.is_continuous:
            mu, log_std = output
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            sample_op = dist.sample().clamp(-1, 1)
        else:
            logits = output
            norm_dist = td.Categorical(logits=logits)
            sample_op = norm_dist.sample()
        return sample_op, cell_state

    def calculate_statistics(self):
        self.data.cal_dc_r(self.gamma, 0., normalize=True)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        def _train(data, cell_state):
            for _ in range(self.epoch):
                loss, entropy = self.train(data, cell_state)
            summaries = dict([
                ['LOSS/loss', loss],
                ['Statistics/entropy', entropy]
            ])
            return summaries

        self._learn(function_dict={
            'calculate_statistics': self.calculate_statistics,
            'train_function': _train,
            'summary_dict': dict([['LEARNING_RATE/lr', self.oplr.lr]])
        })

    @iTensor_oNumpy
    def train(self, BATCH, cell_states):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
        output = self.net(feat)
        if self.is_continuous:
            mu, log_std = output
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            log_act_prob = dist.log_prob(BATCH.action).unsqueeze(-1)
            entropy = dist.entropy().mean()
        else:
            logits = output
            logp_all = logits.log_softmax(-1)
            log_act_prob = (logp_all * BATCH.action).sum(1, keepdim=True)
            entropy = -(logp_all.exp() * logp_all).sum(1, keepdim=True).mean()
        loss = -(log_act_prob * BATCH.discounted_reward).mean()
        self.oplr.step(loss)
        self.global_step.add_(1)
        return loss, entropy
