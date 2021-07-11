#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from torch import distributions as td
from dataclasses import dataclass

from rls.utils.torch_utils import (gaussian_clip_rsample,
                                   gaussian_likelihood_sum,
                                   gaussian_entropy)
from rls.algos.base.on_policy import On_Policy
from rls.utils.specs import (Data,
                             ModelObservations)
from rls.nn.models import (ActorMuLogstd,
                           ActorDct,
                           CriticValue)
from rls.nn.utils import OPLR
from rls.utils.sundry_utils import to_numpy


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
                                       network_settings=network_settings['actor_continuous'])
        else:
            self.actor = ActorDct(self.rep_net.h_dim,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_discrete'])
        self.critic = CriticValue(self.rep_net.h_dim,
                                  network_settings=network_settings['critic'])

        self.actor_op = OPLR(self.actor, actor_lr)
        self.critic_op = OPLR([self.critic, self.rep_net], critic_lr)

        self.initialize_data_buffer(sample_data_type=A2C_Train_BatchExperiences)

        self._worker_modules.update(rep_net=self.rep_net,
                                    actor=self.actor)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(critic=self.critic,
                                     actor_oplr=self.actor_oplr,
                                     critic_oplr=self.critic_oplr)

    def __call__(self, obs, evaluation=False):
        a, self.next_cell_state = self._get_action(obs, self.cell_state)
        return a

    def _get_action(self, obs, cell_state):
        feat, cell_state = self.rep_net(obs.tensor, cell_state=cell_state)
        output = self.actor(feat)
        feat, cell_state = self.rep_net(obs.tensor, cell_state=cell_state)
        if self.is_continuous:
            mu, log_std = output
            sample_op, _ = gaussian_clip_rsample(mu, log_std)
        else:
            logits = output
            norm_dist = td.categorical.Categorical(logits=logits)
            sample_op = norm_dist.sample()
        return to_numpy(sample_op), cell_state

    def _get_value(self, obs, cell_state):
        feat, cell_state = self.rep_net(obs.tensor, cell_state=cell_state)
        value = self.critic(feat)
        return to_numpy(value), cell_state

    def calculate_statistics(self):
        init_value, self.cell_state = self._get_value(self.data.last_data().obs_, cell_state=self.cell_state)
        self.data.cal_dc_r(self.gamma, init_value.numpy())

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        def _train(data, cell_state):
            for _ in range(self.epoch):
                actor_loss, critic_loss, entropy = self.train(data.tensor, cell_state)

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

    def train(self, BATCH, cell_state):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_state['obs'])
        output = self.actor(feat)
        v = self.critic(feat)
        if self.is_continuous:
            mu, log_std = output
            log_act_prob = gaussian_likelihood_sum(BATCH.action, mu, log_std)
            entropy = gaussian_entropy(log_std)
        else:
            logits = output
            logp_all = logits.log_softmax(-1)
            log_act_prob = (BATCH.action * logp_all).sum(1, keepdim=True)
            entropy = -(logp_all.exp() * logp_all).sum(1, keepdim=True).mean()
        advantage = (BATCH.discounted_reward - v).detach()
        td_error = BATCH.discounted_reward - v
        critic_loss = td_error.square().mean()
        actor_loss = -((log_act_prob * advantage).mean() + self.beta * entropy)

        self.actor_oplr.step(actor_loss)
        self.critic_oplr.step(critic_loss)

        self.global_step.add_(1)
        return actor_loss, critic_loss, entropy
