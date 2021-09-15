#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t
from torch import distributions as td

from rls.algorithms.base.sarl_on_policy import SarlOnPolicy
from rls.common.data import Data
from rls.common.decorator import iton
from rls.nn.models import ActorDct, ActorMuLogstd, CriticValue
from rls.nn.utils import OPLR
from rls.utils.np_utils import discounted_sum


class A2C(SarlOnPolicy):
    """
    Synchronous Advantage Actor-Critic, A2C, http://arxiv.org/abs/1602.01783
    """
    policy_mode = 'on-policy'

    def __init__(self,
                 agent_spec,

                 beta=1.0e-3,
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
        super().__init__(agent_spec=agent_spec, **kwargs)
        self.beta = beta

        if self.is_continuous:
            self.actor = ActorMuLogstd(self.obs_spec,
                                       rep_net_params=self._rep_net_params,
                                       output_shape=self.a_dim,
                                       network_settings=network_settings['actor_continuous']).to(self.device)
        else:
            self.actor = ActorDct(self.obs_spec,
                                  rep_net_params=self._rep_net_params,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_discrete']).to(self.device)
        self.critic = CriticValue(self.obs_spec,
                                  rep_net_params=self._rep_net_params,
                                  network_settings=network_settings['critic']).to(self.device)

        self.actor_oplr = OPLR(self.actor, actor_lr, **self._oplr_params)
        self.critic_oplr = OPLR(self.critic, critic_lr, **self._oplr_params)

        self._trainer_modules.update(actor=self.actor,
                                     critic=self.critic,
                                     actor_oplr=self.actor_oplr,
                                     critic_oplr=self.critic_oplr)

    @iton
    def select_action(self, obs):
        output = self.actor(obs, rnncs=self.rnncs)    # [B, A]
        self.rnncs_ = self.actor.get_rnncs()
        if self.is_continuous:
            mu, log_std = output     # [B, A]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            action = dist.sample().clamp(-1, 1)   # [B, A]
        else:
            logits = output  # [B, A]
            norm_dist = td.Categorical(logits=logits)
            action = norm_dist.sample()   # [B,]

        acts_info = Data(action=action)
        if self.use_rnn:
            acts_info.update(rnncs=self.rnncs)
        return action, acts_info

    @iton
    def _get_value(self, obs, rnncs=None):
        value = self.critic(obs, rnncs=self.rnncs)
        return value

    def _preprocess_BATCH(self, BATCH):  # [T, B, *]
        BATCH = super()._preprocess_BATCH(BATCH)
        value = self._get_value(BATCH.obs_[-1], rnncs=self.rnncs)
        BATCH.discounted_reward = discounted_sum(BATCH.reward,
                                                 self.gamma,
                                                 BATCH.done,
                                                 BATCH.begin_mask,
                                                 init_value=value)
        td_error = calculate_td_error(BATCH.reward,
                                      self.gamma,
                                      BATCH.done,
                                      value=BATCH.value,
                                      next_value=np.concatenate((BATCH.value[1:], value[np.newaxis, :]), 0))
        BATCH.gae_adv = discounted_sum(td_error,
                                       self.lambda_*self.gamma,
                                       BATCH.done,
                                       BATCH.begin_mask,
                                       init_value=0.,
                                       normalize=True)
        return BATCH

    @iton
    def _train(self, BATCH):
        v = self.critic(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, 1]
        td_error = BATCH.discounted_reward - v   # [T, B, 1]
        critic_loss = td_error.square().mean()  # 1
        self.critic_oplr.optimize(critic_loss)

        if self.is_continuous:
            mu, log_std = self.actor(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            log_act_prob = dist.log_prob(BATCH.action).unsqueeze(-1)     # [T, B, 1]
            entropy = dist.entropy().unsqueeze(-1)     # [T, B, 1]
        else:
            logits = self.actor(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
            logp_all = logits.log_softmax(-1)   # [T, B, A]
            log_act_prob = (BATCH.action * logp_all).sum(-1, keepdim=True)  # [T, B, 1]
            entropy = -(logp_all.exp() * logp_all).sum(-1, keepdim=True)  # [T, B, 1]
        advantage = BATCH.discounted_reward - v.detach()    # [T, B, 1]
        actor_loss = -(log_act_prob * BATCH.gae_adv + self.beta * entropy).mean()  # 1
        self.actor_oplr.optimize(actor_loss)

        return dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/critic_loss', critic_loss],
            ['Statistics/entropy', entropy.mean()],
            ['LEARNING_RATE/actor_lr', self.actor_oplr.lr],
            ['LEARNING_RATE/critic_lr', self.critic_oplr.lr]
        ])
