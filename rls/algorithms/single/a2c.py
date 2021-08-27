#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t
from torch import distributions as td

from rls.algorithms.base.sarl_on_policy import SarlOnPolicy
from rls.common.decorator import iTensor_oNumpy
from rls.common.specs import Data
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

        self.actor_oplr = OPLR(self.actor, actor_lr)
        self.critic_oplr = OPLR(self.critic, critic_lr)

        self._trainer_modules.update(actor=self.actor,
                                     critic=self.critic,
                                     actor_oplr=self.actor_oplr,
                                     critic_oplr=self.critic_oplr)

    @iTensor_oNumpy
    def select_action(self, obs):
        output = self.actor(obs, cell_state=self.cell_state)    # [B, A]
        self.next_cell_state = self.actor.get_cell_state()
        if self.is_continuous:
            mu, log_std = output     # [B, A]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            action = dist.sample().clamp(-1, 1)   # [B, A]
        else:
            logits = output  # [B, A]
            norm_dist = td.Categorical(logits=logits)
            action = norm_dist.sample()   # [B,]

        acts = Data(action=action)
        if self.use_rnn:
            acts.update(cell_state=self.cell_state)
        return action, acts

    @iTensor_oNumpy
    def _get_value(self, obs):
        value = self.critic(obs)
        return value

    def _preprocess_BATCH(self, BATCH):  # [T, B, *]
        BATCH = super()._preprocess_BATCH(BATCH)
        value = self._get_value(BATCH.obs_[-1])
        BATCH.discounted_reward = discounted_sum(BATCH.reward,
                                                 self.gamma,
                                                 BATCH.done,
                                                 BATCH.begin_mask,
                                                 init_value=value)
        return BATCH

    @iTensor_oNumpy
    def _train(self, BATCH):
        v = self.critic(BATCH.obs)  # [T, B, 1]
        td_error = BATCH.discounted_reward - v   # [T, B, 1]
        critic_loss = td_error.square().mean()  # 1
        self.critic_oplr.step(critic_loss)

        if self.is_continuous:
            mu, log_std = self.actor(BATCH.obs)  # [T, B, A]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            log_act_prob = dist.log_prob(
                BATCH.action).unsqueeze(-1)     # [T, B, 1]
            entropy = dist.entropy().unsqueeze(-1)     # [T, B, 1]
        else:
            logits = self.actor(BATCH.obs)  # [T, B, A]
            logp_all = logits.log_softmax(-1)   # [T, B, A]
            log_act_prob = (BATCH.action * logp_all).sum(-1,
                                                         keepdim=True)  # [T, B, 1]
            entropy = -(logp_all.exp() * logp_all).sum(-1,
                                                       keepdim=True)  # [T, B, 1]
        advantage = BATCH.discounted_reward - v.detach()    # [T, B, 1]
        actor_loss = -(log_act_prob * advantage +
                       self.beta * entropy).mean()  # 1
        self.actor_oplr.step(actor_loss)

        return dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/critic_loss', critic_loss],
            ['Statistics/entropy', entropy.mean()],
            ['LEARNING_RATE/actor_lr', self.actor_oplr.lr],
            ['LEARNING_RATE/critic_lr', self.critic_oplr.lr]
        ])
