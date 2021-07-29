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
                           ActorDct,
                           CriticValue)
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


@dataclass(eq=False)
class A2C_Train_BatchExperiences(Data):
    obs: ModelObservations
    action: np.ndarray
    discounted_reward: np.ndarray


class A2C(On_Policy):
    def __init__(self,
                 envspec,

                 epoch=5,
                 beta=1.0e-3,
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 network_settings={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'critic': [32, 32]
                 },
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.beta = beta
        self.epoch = epoch

        if self.is_continuous:
            self.actor = ActorMuLogstd(self.rep_net.h_dim,
                                       output_shape=self.a_dim,
                                       network_settings=network_settings['actor_continuous']).to(self.device)
        else:
            self.actor = ActorDct(self.rep_net.h_dim,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_discrete']).to(self.device)
        self.critic = CriticValue(self.rep_net.h_dim,
                                  network_settings=network_settings['critic']).to(self.device)

        self.actor_op = OPLR(self.actor, actor_lr)
        self.critic_op = OPLR([self.critic, self.rep_net], critic_lr)

        self.initialize_data_buffer(sample_data_type=A2C_Train_BatchExperiences)

        self._worker_modules.update(rep_net=self.rep_net,
                                    actor=self.actor)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(critic=self.critic,
                                     actor_oplr=self.actor_oplr,
                                     critic_oplr=self.critic_oplr)

    @iTensor_oNumpy
    def __call__(self, obs, evaluation=False):
        feat, self.next_cell_state = self.rep_net(obs, cell_state=self.cell_state)
        output = self.actor(feat)
        if self.is_continuous:
            mu, log_std = output
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            sample_op = dist.sample().clamp(-1, 1)
        else:
            logits = output
            norm_dist = td.Categorical(logits=logits)
            sample_op = norm_dist.sample()
        return sample_op

    @iTensor_oNumpy
    def _get_value(self, obs):
        feat, _ = self.rep_net(obs, cell_state=self.cell_state)
        value = self.critic(feat)
        return value

    def calculate_statistics(self):
        init_value = self._get_value(self.data.get_last_date().obs_, cell_state=)
        self.data.cal_dc_r(self.gamma, init_value)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        def _train(data, cell_state):
            for _ in range(self.epoch):
                actor_loss, critic_loss, entropy = self.train(data, cell_state)

            summaries = dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/entropy', entropy],
            ])
            return summaries

        self._learn(function_dict={
            'calculate_statistics': self.calculate_statistics,
            'train_function': _train,
            'summary_dict': dict([
                ['LEARNING_RATE/actor_lr', self.actor_oplr.lr],
                ['LEARNING_RATE/critic_lr', self.critic_oplr.lr]
            ])
        })

    @iTensor_oNumpy
    def train(self, BATCH, cell_states):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
        v = self.critic(feat)
        td_error = BATCH.discounted_reward - v
        critic_loss = td_error.square().mean()
        self.critic_oplr.step(critic_loss)

        feat = feat.detach()
        if self.is_continuous:
            mu, log_std = self.actor(feat)
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            log_act_prob = dist.log_prob(BATCH.action).unsqueeze(-1)
            entropy = dist.entropy().mean()
        else:
            logits = self.actor(feat)
            logp_all = logits.log_softmax(-1)
            log_act_prob = (BATCH.action * logp_all).sum(1, keepdim=True)
            entropy = -(logp_all.exp() * logp_all).sum(1, keepdim=True).mean()
        advantage = BATCH.discounted_reward - v.detach()
        actor_loss = -((log_act_prob * advantage).mean() + self.beta * entropy)
        self.actor_oplr.step(actor_loss)

        self.global_step.add_(1)
        return actor_loss, critic_loss, entropy
