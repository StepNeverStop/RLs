#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from torch import distributions as td
from dataclasses import dataclass

from rls.utils.torch_utils import (gaussian_clip_rsample,
                                   gaussian_likelihood_sum,
                                   q_target_func,
                                   gaussian_entropy)
from rls.algos.base.off_policy import Off_Policy
from rls.common.specs import BatchExperiences
from rls.nn.models import (ActorMuLogstd,
                           ActorDct,
                           CriticQvalueOne)
from rls.nn.utils import OPLR
from rls.utils.converter import to_numpy
from rls.common.decorator import iTensor_oNumpy


@dataclass(eq=False)
class AC_BatchExperiences(BatchExperiences):
    old_log_prob: np.ndarray


class AC(Off_Policy):
    # off-policy actor-critic
    def __init__(self,
                 envspec,

                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 network_settings={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'critic': [32, 32]
                 },
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)

        if self.is_continuous:
            self.actor = ActorMuLogstd(self.rep_net.h_dim,
                                       output_shape=self.a_dim,
                                       network_settings=network_settings['actor_continuous']).to(self.device)
        else:
            self.actor = ActorDct(self.rep_net.h_dim,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_discrete']).to(self.device)
        self.critic = CriticQvalueOne(self.rep_net.h_dim,
                                      action_dim=self.a_dim,
                                      network_settings=network_settings['critic']).to(self.device)

        self.actor_oplr = OPLR(self.actor, actor_lr)
        self.critic_oplr = OPLR([self.critic, self.rep_net], critic_lr)

        self._worker_modules.update(rep_net=self.rep_net,
                                    actor=self.actor)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(critic=self.critic,
                                     actor_oplr=self.actor_oplr,
                                     critic_oplr=self.critic_oplr)
        self.initialize_data_buffer()

    @iTensor_oNumpy
    def __call__(self, obs, evaluation=False):
        """
        choose an action according to a given observation
        :param obs: 
        :param evaluation:
        """
        feat, self.cell_state = self.rep_net(obs, cell_state=self.cell_state)
        output = self.actor(feat)
        if self.is_continuous:
            mu, log_std = output
            sample_op, _ = gaussian_clip_rsample(mu, log_std)
            log_prob = gaussian_likelihood_sum(sample_op, mu, log_std)
        else:
            logits = output
            norm_dist = td.categorical.Categorical(logits=logits)
            sample_op = norm_dist.sample()
            log_prob = norm_dist.log_prob(sample_op)
        self._log_prob = to_numpy(log_prob)
        return sample_op

    def store_data(self, exps: BatchExperiences):
        # self._running_average()
        self.data.add(AC_BatchExperiences(*exps.astuple(), self._log_prob))

    def no_op_store(self, exps: BatchExperiences):
        # self._running_average()
        self.data.add(AC_BatchExperiences(*exps.astuple(), np.ones_like(exps.reward)))

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([
                    ['LEARNING_RATE/actor_lr', self.actor_oplr.lr],
                    ['LEARNING_RATE/critic_lr', self.critic_oplr.lr]
                ])
            })

    @iTensor_oNumpy
    def _train(self, BATCH, isw, cell_states):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
        output = self.actor(feat)
        q = self.critic(feat)
        feat_, _ = self.rep_net(BATCH.obs_, cell_state=cell_states['obs_'])
        if self.is_continuous:
            mu, log_std = output
            log_prob = gaussian_likelihood_sum(BATCH.action, mu, log_std)
            entropy = gaussian_entropy(log_std)

            next_mu, _ = self.actor(feat_)
            max_q_next = self.critic(feat_, next_mu).detach()
        else:
            logits = output
            logp_all = logits.log_softmax(-1)
            log_prob = (logp_all * BATCH.action).sum(1, keepdim=True)
            entropy = -(logp_all.exp() * logp_all).sum(1, keepdim=True).mean()

            logits = self.actor(feat_)
            max_a = logits.argmax(1)
            max_a_one_hot = t.nn.functional.one_hot(max_a, self.a_dim).float()
            max_q_next = self.critic(feat_, max_a_one_hot).detach()
        ratio = (log_prob - BATCH.old_log_prob).exp().detach()
        q_value = q.detach()
        td_error = q_target_func(BATCH.reward,
                                 self.gamma,
                                 BATCH.done,
                                 max_q_next)
        critic_loss = (td_error.square() * isw).mean()
        actor_loss = -(ratio * log_prob * q_value).mean()

        self.actor_oplr.step(actor_loss)
        self.critic_oplr.step(critic_loss)

        self.global_step.add_(1)
        return td_error, dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/critic_loss', critic_loss],
            ['Statistics/q_max', q.max()],
            ['Statistics/q_min', q.min()],
            ['Statistics/q_mean', q.mean()],
            ['Statistics/ratio', ratio.mean()],
            ['Statistics/entropy', entropy]
        ])
