#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from torch import distributions as td

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.utils.torch_utils import q_target_func
from rls.nn.noised_actions import Noise_action_REGISTER
from rls.nn.models import (ActorDPG,
                           ActorDct,
                           CriticQvalueOne)
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy
from rls.nn.modules.wrappers import TargetTwin
from rls.common.specs import Data


class TD3(SarlOffPolicy):
    '''
    Twin Delayed Deep Deterministic Policy Gradient, https://arxiv.org/abs/1802.09477
    '''
    policy_mode = 'off-policy'

    def __init__(self,
                 ployak=0.995,
                 delay_num=2,
                 noise_action='clip_normal',
                 noise_params={
                     'sigma': 0.2,
                     'noise_bound': 0.2
                 },
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 discrete_tau=1.0,
                 network_settings={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'q': [32, 32]
                 },
                 **kwargs):
        super().__init__(**kwargs)
        self.ployak = ployak
        self.delay_num = delay_num
        self.discrete_tau = discrete_tau

        if self.is_continuous:
            actor = ActorDPG(self.obs_spec,
                             rep_net_params=self.rep_net_params,
                             output_shape=self.a_dim,
                             network_settings=network_settings['actor_continuous'])
            self.noised_action = self.target_noised_action = Noise_action_REGISTER[noise_action](**noise_params)
        else:
            actor = ActorDct(self.obs_spec,
                             rep_net_params=self.rep_net_params,
                             output_shape=self.a_dim,
                             network_settings=network_settings['actor_continuous'])
        self.actor = TargetTwin(actor, self.ployak).to(self.device)

        self.critic = TargetTwin(CriticQvalueOne(self.obs_spec,
                                                 rep_net_params=self.rep_net_params,
                                                 action_dim=self.a_dim,
                                                 network_settings=network_settings['q']),
                                 self.ployak).to(self.device)
        self.critic2 = deepcopy(self.critic)

        self.actor_oplr = OPLR(self.actor, actor_lr)
        self.critic_oplr = OPLR([self.critic, self.critic2],  critic_lr)
        self._trainer_modules.update(actor=self.actor,
                                     critic=self.critic,
                                     critic2=self.critic2,
                                     actor_oplr=self.actor_oplr,
                                     critic_oplr=self.critic_oplr)

    def episode_reset(self):
        super().episode_reset()
        if self.is_continuous:
            self.noised_action.reset()

    @iTensor_oNumpy
    def __call__(self, obs):
        output = self.actor(obs, cell_state=self.cell_state)    # [B, A]
        self.next_cell_state = self.actor.get_cell_state()
        if self.is_continuous:
            mu = output  # [B, A]
            pi = self.noised_action(mu)  # [B, A]
        else:
            logits = output  # [B, A]
            mu = logits.argmax(-1)   # [B,]
            cate_dist = td.Categorical(logits=logits)
            pi = cate_dist.sample()  # [B,]
        actions = pi if self._is_train_mode else mu
        return Data(action=actions)

    @iTensor_oNumpy
    def _train(self, BATCH):
        for _ in range(self.delay_num):
            if self.is_continuous:
                action_target = self.target_noised_action(self.actor.t(BATCH.obs_))  # [T, B, A]
            else:
                target_logits = self.actor.t(BATCH.obs_)    # [T, B, A]
                target_cate_dist = td.Categorical(logits=target_logits)
                target_pi = target_cate_dist.sample()   # [T, B]
                action_target = t.nn.functional.one_hot(target_pi, self.a_dim).float()  # [T, B, A]
            q1 = self.critic(BATCH.obs, BATCH.action)   # [T, B, 1]
            q2 = self.critic2(BATCH.obs, BATCH.action)   # [T, B, 1]
            q_target = t.minimum(self.critic.t(BATCH.obs_, action_target), self.critic2.t(BATCH.obs_, action_target))    # [T, B, 1]
            dc_r = q_target_func(BATCH.reward,
                                 self.gamma,
                                 BATCH.done,
                                 q_target,
                                 BATCH.begin_mask,
                                 use_rnn=self.use_rnn)   # [T, B, 1]
            td_error1 = q1 - dc_r    # [T, B, 1]
            td_error2 = q2 - dc_r    # [T, B, 1]

            q1_loss = (td_error1.square() * BATCH.get('isw', 1.0)).mean()    # 1
            q2_loss = (td_error2.square() * BATCH.get('isw', 1.0)).mean()    # 1
            critic_loss = 0.5 * (q1_loss + q2_loss)
            self.critic_oplr.step(critic_loss)

        if self.is_continuous:
            mu = self.actor(BATCH.obs)  # [T, B, A]
        else:
            logits = self.actor(BATCH.obs)  # [T, B, A]
            logp_all = logits.log_softmax(-1)   # [T, B, A]
            gumbel_noise = td.Gumbel(0, 1).sample(logp_all.shape)   # [T, B, A]
            _pi = ((logp_all + gumbel_noise) / self.discrete_tau).softmax(-1)   # [T, B, A]
            _pi_true_one_hot = t.nn.functional.one_hot(_pi.argmax(-1), self.a_dim).float()  # [T, B, A]
            _pi_diff = (_pi_true_one_hot - _pi).detach()    # [T, B, A]
            mu = _pi_diff + _pi  # [T, B, A]
        q1_actor = self.critic(BATCH.obs, mu)   # [T, B, 1]

        actor_loss = -q1_actor.mean()   # 1
        self.actor_oplr.step(actor_loss)
        return (td_error1 + td_error2) / 2, dict([
            ['LEARNING_RATE/actor_lr', self.actor_oplr.lr],
            ['LEARNING_RATE/critic_lr', self.critic_oplr.lr],
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/critic_loss', critic_loss],
            ['Statistics/q_min', t.minimum(q1, q2).min()],
            ['Statistics/q_mean', t.minimum(q1, q2).mean()],
            ['Statistics/q_max', t.maximum(q1, q2).max()]
        ])

    def _after_train(self):
        super()._after_train()
        self.actor.sync()
        self.critic.sync()
        self.critic2.sync()
