#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t
import torch.nn.functional as F
from torch import distributions as td

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.common.decorator import iton
from rls.common.specs import Data
from rls.nn.models import ActorDct, ActorDPG, CriticQvalueOne
from rls.nn.noised_actions import (ClippedNormalNoisedAction,
                                   Noise_action_REGISTER)
from rls.nn.utils import OPLR
from rls.utils.torch_utils import n_step_return


class DPG(SarlOffPolicy):
    '''
    Deterministic Policy Gradient, https://hal.inria.fr/file/index/docid/938992/filename/dpg-icml2014.pdf
    '''
    policy_mode = 'off-policy'

    def __init__(self,
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 use_target_action_noise=False,
                 noise_action='ou',
                 noise_params={
                     'sigma': 0.2
                 },
                 discrete_tau=1.0,
                 network_settings={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'q': [32, 32]
                 },
                 **kwargs):
        super().__init__(**kwargs)
        self.discrete_tau = discrete_tau
        self.use_target_action_noise = use_target_action_noise

        if self.is_continuous:
            self.target_noised_action = ClippedNormalNoisedAction(sigma=0.2, noise_bound=0.2)
            self.noised_action = Noise_action_REGISTER[noise_action](**noise_params)
            self.actor = ActorDPG(self.obs_spec,
                                  rep_net_params=self._rep_net_params,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_continuous']).to(self.device)
        else:
            self.actor = ActorDct(self.obs_spec,
                                  rep_net_params=self._rep_net_params,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_discrete']).to(self.device)

        self.critic = CriticQvalueOne(self.obs_spec,
                                      rep_net_params=self._rep_net_params,
                                      action_dim=self.a_dim,
                                      network_settings=network_settings['q']).to(self.device)

        self.actor_oplr = OPLR(self.actor, actor_lr, **self._oplr_params)
        self.critic_oplr = OPLR(self.critic, critic_lr, **self._oplr_params)
        self._trainer_modules.update(actor=self.actor,
                                     critic=self.critic,
                                     actor_oplr=self.actor_oplr,
                                     critic_oplr=self.critic_oplr)

    def episode_reset(self):
        super().episode_reset()
        if self.is_continuous:
            self.noised_action.reset()

    @iton
    def select_action(self, obs):
        output = self.actor(obs, rnncs=self.rnncs)    # [B, A]
        self.rnncs_ = self.actor.get_rnncs()
        if self.is_continuous:
            mu = output  # [B, A]
            pi = self.noised_action(mu)  # [B, A]
        else:
            logits = output  # [B, A]
            mu = logits.argmax(-1)   # [B,]
            cate_dist = td.Categorical(logits=logits)
            pi = cate_dist.sample()  # [B,]
        actions = pi if self._is_train_mode else mu
        return actions, Data(action=actions)

    @iton
    def _train(self, BATCH):
        if self.is_continuous:
            action_target = self.actor(BATCH.obs_, begin_mask=BATCH.begin_mask)  # [T, B, A]
            if self.use_target_action_noise:
                action_target = self.target_noised_action(action_target)    # [T, B, A]
        else:
            target_logits = self.actor(BATCH.obs_, begin_mask=BATCH.begin_mask)  # [T, B, A]
            target_cate_dist = td.Categorical(logits=target_logits)
            target_pi = target_cate_dist.sample()   # [T, B]
            action_target = F.one_hot(target_pi, self.a_dim).float()  # [T, B, A]
        q_target = self.critic(BATCH.obs_, action_target,                               begin_mask=BATCH.begin_mask)   # [T, B, 1]
        dc_r = n_step_return(BATCH.reward,
                             self.gamma,
                             BATCH.done,
                             q_target,
                             BATCH.begin_mask).detach()  # [T, B, 1]
        q = self.critic(BATCH.obs, BATCH.action,                        begin_mask=BATCH.begin_mask)    # [T, B, A]
        td_error = dc_r - q  # [T, B, A]
        q_loss = (td_error.square()*BATCH.get('isw', 1.0)).mean()   # 1
        self.critic_oplr.optimize(q_loss)

        if self.is_continuous:
            mu = self.actor(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
        else:
            logits = self.actor(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
            _pi = logits.softmax(-1)    # [T, B, A]
            _pi_true_one_hot = F.one_hot(
                logits.argmax(-1), self.a_dim).float()   # [T, B, A]
            _pi_diff = (_pi_true_one_hot - _pi).detach()    # [T, B, A]
            mu = _pi_diff + _pi  # [T, B, A]
        q_actor = self.critic(BATCH.obs, mu, begin_mask=BATCH.begin_mask)    # [T, B, 1]
        actor_loss = -q_actor.mean()   # 1
        self.actor_oplr.optimize(actor_loss)

        return td_error, dict([
            ['LEARNING_RATE/actor_lr', self.actor_oplr.lr],
            ['LEARNING_RATE/critic_lr', self.critic_oplr.lr],
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/critic_loss', q_loss],
            ['Statistics/q_min', q.min()],
            ['Statistics/q_mean', q.mean()],
            ['Statistics/q_max', q.max()]
        ])
