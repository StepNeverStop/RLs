#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from torch import distributions as td

from rls.algorithms.base.sarl_on_policy import SarlOnPolicy
from rls.common.specs import Data
from rls.nn.models import (ActorMuLogstd,
                           ActorDct)
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy
from rls.utils.np_utils import discounted_sum


class PG(SarlOnPolicy):
    policy_mode = 'on-policy'

    def __init__(self,
                 agent_spec,

                 lr=5.0e-4,
                 network_settings={
                     'actor_continuous': {
                         'hidden_units': [32, 32],
                         'condition_sigma': False,
                         'log_std_bound': [-20, 2]
                     },
                     'actor_discrete': [32, 32]
                 },
                 **kwargs):
        super().__init__(agent_spec=agent_spec, **kwargs)
        if self.is_continuous:
            self.net = ActorMuLogstd(self.obs_spec,
                                     rep_net_params=self.rep_net_params,
                                     output_shape=self.a_dim,
                                     network_settings=network_settings['actor_continuous']).to(self.device)
        else:
            self.net = ActorDct(self.obs_spec,
                                rep_net_params=self.rep_net_params,
                                output_shape=self.a_dim,
                                network_settings=network_settings['actor_discrete']).to(self.device)
        self.oplr = OPLR(self.net, lr)

        self._trainer_modules.update(model=self.net,
                                     oplr=self.oplr)

    @iTensor_oNumpy
    def select_action(self, obs):
        output = self.net(obs, cell_state=self.cell_state)  # [B, A]
        self.next_cell_state = self.net.get_cell_state()
        if self.is_continuous:
            mu, log_std = output    # [B, A]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            action = dist.sample().clamp(-1, 1)  # [B, A]
        else:
            logits = output  # [B, A]
            norm_dist = td.Categorical(logits=logits)
            action = norm_dist.sample()  # [B,]

        acts = Data(action=action)
        if self.use_rnn:
            acts.update(cell_state=self.cell_state)
        return action, acts

    def _preprocess_BATCH(self, BATCH):  # [T, B, *]
        BATCH = super()._preprocess_BATCH(BATCH)
        BATCH.discounted_reward = discounted_sum(BATCH.reward,
                                                 self.gamma,
                                                 BATCH.done,
                                                 BATCH.begin_mask,
                                                 init_value=0.,
                                                 normalize=True)
        return BATCH

    @iTensor_oNumpy
    def _train(self, BATCH):     # [B, T, *]
        output = self.net(BATCH.obs)    # [B, T, A]
        if self.is_continuous:
            mu, log_std = output    # [B, T, A]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            log_act_prob = dist.log_prob(
                BATCH.action).unsqueeze(-1)    # [B, T, 1]
            entropy = dist.entropy().unsqueeze(-1)  # [B, T, 1]
        else:
            logits = output  # [B, T, A]
            logp_all = logits.log_softmax(-1)   # [B, T, A]
            log_act_prob = (logp_all * BATCH.action).sum(-1,
                                                         keepdim=True)  # [B, T, 1]
            entropy = -(logp_all.exp() * logp_all).sum(1,
                                                       keepdim=True)  # [B, T, 1]
        loss = -(log_act_prob * BATCH.discounted_reward).mean()
        self.oplr.step(loss)
        return dict([
            ['LOSS/loss', loss],
            ['Statistics/entropy', entropy.mean()],
            ['LEARNING_RATE/lr', self.oplr.lr]
        ])
