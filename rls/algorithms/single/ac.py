#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from torch import distributions as td

from rls.utils.torch_utils import q_target_func
from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.common.specs import Data
from rls.nn.models import (ActorMuLogstd,
                           ActorDct,
                           CriticQvalueOne)
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


class AC(SarlOffPolicy):
    policy_mode = 'off-policy'

    # off-policy actor-critic
    def __init__(self,
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 network_settings={
                     'actor_continuous': {
                         'hidden_units': [64, 64],
                         'condition_sigma': False,
                         'log_std_bound': [-20, 2]
                     },
                     'actor_discrete': [32, 32],
                     'critic': [32, 32]
                 },
                 **kwargs):
        super().__init__(**kwargs)

        if self.is_continuous:
            self.actor = ActorMuLogstd(self.obs_spec,
                                       rep_net_params=self.rep_net_params,
                                       output_shape=self.a_dim,
                                       network_settings=network_settings['actor_continuous']).to(self.device)
        else:
            self.actor = ActorDct(self.obs_spec,
                                  rep_net_params=self.rep_net_params,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_discrete']).to(self.device)
        self.critic = CriticQvalueOne(self.obs_spec,
                                      rep_net_params=self.rep_net_params,
                                      action_dim=self.a_dim,
                                      network_settings=network_settings['critic']).to(self.device)

        self.actor_oplr = OPLR(self.actor, actor_lr)
        self.critic_oplr = OPLR(self.critic, critic_lr)

        self._trainer_modules.update(actor=self.actor,
                                     critic=self.critic,
                                     actor_oplr=self.actor_oplr,
                                     critic_oplr=self.critic_oplr)

    @iTensor_oNumpy
    def select_action(self, obs):
        output = self.actor(obs, cell_state=self.cell_state)    # [B, *]
        self.next_cell_state = self.actor.get_cell_state()
        if self.is_continuous:
            mu, log_std = output    # [B, *]
            dist = td.Independent(td.Normal(mu, log_std.exp()), -1)
            action = dist.sample().clamp(-1, 1)   # [B, *]
            log_prob = dist.log_prob(action)    # [B,]
        else:
            logits = output  # [B, *]
            norm_dist = td.Categorical(logits=logits)
            action = norm_dist.sample()   # [B,]
            log_prob = norm_dist.log_prob(action)  # [B,]
        return action, Data(action=action,
                            log_prob=log_prob)

    def random_action(self):
        acts = super().random_action()
        if self.is_continuous:
            acts.update(log_prob=np.full(self.n_copys, np.log(0.5)))  # [B,]
        else:
            acts.update(log_prob=np.full(self.n_copys, 1./self.a_dim))  # [B,]
        return acts

    @iTensor_oNumpy
    def _train(self, BATCH):
        q = self.critic(BATCH.obs, BATCH.action)    # [T, B, 1]
        if self.is_continuous:
            next_mu, _ = self.actor(BATCH.obs_)  # [T, B, *]
            max_q_next = self.critic(BATCH.obs_, next_mu).detach()  # [T, B, 1]
        else:
            logits = self.actor(BATCH.obs_)  # [T, B, *]
            max_a = logits.argmax(-1)    # [T, B]
            max_a_one_hot = t.nn.functional.one_hot(
                max_a, self.a_dim).float()  # [T, B, N]
            max_q_next = self.critic(
                BATCH.obs_, max_a_one_hot).detach()    # [T, B, 1]
        td_error = q - q_target_func(BATCH.reward,
                                     self.gamma,
                                     BATCH.done,
                                     max_q_next,
                                     BATCH.begin_mask,
                                     use_rnn=self.use_rnn)  # [T, B, 1]
        critic_loss = (td_error.square()*BATCH.get('isw', 1.0)).mean()   # 1
        self.critic_oplr.step(critic_loss)

        if self.is_continuous:
            mu, log_std = self.actor(BATCH.obs)  # [T, B, *]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            log_prob = dist.log_prob(BATCH.action)    # [T, B]
            entropy = dist.entropy().mean()  # 1
        else:
            logits = self.actor(BATCH.obs)  # [T, B, *]
            logp_all = logits.log_softmax(-1)   # [T, B, *]
            log_prob = (logp_all * BATCH.action).sum(-1)  # [T, B]
            entropy = -(logp_all.exp() * logp_all).sum(-1).mean()   # 1
        ratio = (log_prob - BATCH.log_prob).exp().detach()  # [T, B]
        actor_loss = -(ratio * log_prob * q.squeeze(-1).detach()
                       ).mean()    # [T, B] => 1
        self.actor_oplr.step(actor_loss)

        return td_error, dict([
            ['LEARNING_RATE/actor_lr', self.actor_oplr.lr],
            ['LEARNING_RATE/critic_lr', self.critic_oplr.lr],
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/critic_loss', critic_loss],
            ['Statistics/q_max', q.max()],
            ['Statistics/q_min', q.min()],
            ['Statistics/q_mean', q.mean()],
            ['Statistics/ratio', ratio.mean()],
            ['Statistics/entropy', entropy]
        ])
