#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from torch import distributions as td

from rls.algorithms.base.off_policy import Off_Policy
from rls.utils.torch_utils import (sync_params_pairs,
                                   q_target_func)
from rls.nn.noised_actions import Noise_action_REGISTER
from rls.nn.models import (ActorDPG,
                           ActorDct,
                           CriticQvalueOne)
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


class TD3(Off_Policy):
    '''
    Twin Delayed Deep Deterministic Policy Gradient, https://arxiv.org/abs/1802.09477
    '''

    def __init__(self,
                 envspec,

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
        super().__init__(envspec=envspec, **kwargs)
        self.ployak = ployak
        self.delay_num = delay_num
        self.discrete_tau = discrete_tau

        if self.is_continuous:
            self.actor = ActorDPG(self.rep_net.h_dim,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_continuous']).to(self.device)
            self.noised_action = self.target_noised_action = Noise_action_REGISTER[noise_action](**noise_params)
        else:
            self.actor = ActorDct(self.rep_net.h_dim,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_continuous']).to(self.device)

        self.actor_target = deepcopy(self.actor)
        self.actor_target.eval()

        self.critic = CriticQvalueOne(self.rep_net.h_dim,
                                      action_dim=self.a_dim,
                                      network_settings=network_settings['q']).to(self.device)
        self.critic2 = CriticQvalueOne(self.rep_net.h_dim,
                                       action_dim=self.a_dim,
                                       network_settings=network_settings['q']).to(self.device)

        self.critic_target = deepcopy(self.critic)
        self.critic_target.eval()
        self.critic2_target = deepcopy(self.critic2)
        self.critic2_target.eval()
        self._target_rep_net = deepcopy(self.rep_net)
        self._target_rep_net.eval()

        self._pairs = [(self._target_rep_net, self.rep_net),
                       (self.critic_target, self.critic),
                       (self.critic2_target, self.critic2),
                       (self.actor_target, self.actor)]
        sync_params_pairs(self._pairs)

        self.actor_oplr = OPLR(self.actor, actor_lr)
        self.critic_oplr = OPLR([self.critic, self.critic2, self.rep_net],  critic_lr)

        self._worker_modules.update(rep_net=self.rep_net,
                                    actor=self.actor)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(critic=self.critic,
                                     critic2=self.critic2,
                                     actor_oplr=self.actor_oplr,
                                     critic_oplr=self.critic_oplr)
        self.initialize_data_buffer()

    def reset(self):
        super().reset()
        if self.is_continuous:
            self.noised_action.reset()

    @iTensor_oNumpy
    def __call__(self, obs, evaluation=False):
        feat, self.cell_state = self.rep_net(obs, cell_state=self.cell_state)
        output = self.actor(feat)
        if self.is_continuous:
            mu = output
            pi = self.noised_action(mu)
        else:
            logits = output
            mu = logits.argmax(1)
            cate_dist = td.Categorical(logits=logits)
            pi = cate_dist.sample()
        return mu if evaluation else pi

    def _target_params_update(self):
        sync_params_pairs(self._pairs, self.ployak)

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
        for _ in range(self.delay_num):
            feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
            feat_, _ = self._target_rep_net(BATCH.obs_, cell_state=cell_states['obs_'])
            if self.is_continuous:
                action_target = self.target_noised_action(self.actor_target(feat_))
            else:
                target_logits = self.actor_target(feat_)
                target_cate_dist = td.Categorical(logits=target_logits)
                target_pi = target_cate_dist.sample()
                target_log_pi = target_cate_dist.log_prob(target_pi)
                action_target = t.nn.functional.one_hot(target_pi, self.a_dim).float()
            q1 = self.critic(feat, BATCH.action)
            q2 = self.critic2(feat, BATCH.action)
            q_target = t.minimum(self.critic_target(feat_, action_target), self.critic2_target(feat_, action_target))
            dc_r = q_target_func(BATCH.reward,
                                self.gamma,
                                BATCH.done,
                                q_target)
            td_error1 = q1 - dc_r
            td_error2 = q2 - dc_r
            q1_loss = (td_error1.square() * isw).mean()
            q2_loss = (td_error2.square() * isw).mean()
            critic_loss = 0.5 * (q1_loss + q2_loss)
            self.critic_oplr.step(critic_loss)

        feat = feat.detach()
        if self.is_continuous:
            mu = self.actor(feat)
        else:
            logits = self.actor(feat)
            logp_all = logits.log_softmax(-1)
            gumbel_noise = td.Gumbel(0, 1).sample(BATCH.action.shape)
            _pi = ((logp_all + gumbel_noise) / self.discrete_tau).softmax(-1)
            _pi_true_one_hot = t.nn.functional.one_hot(_pi.argmax(-1), self.a_dim).float()
            _pi_diff = (_pi_true_one_hot - _pi).detach()
            mu = _pi_diff + _pi
        q1_actor = self.critic(feat, mu)
        actor_loss = -q1_actor.mean()
        self.actor_oplr.step(actor_loss)
        self.global_step.add_(1)
        return (td_error1 + td_error2) / 2, dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/critic_loss', critic_loss],
            ['Statistics/q_min', t.minimum(q1, q2).min()],
            ['Statistics/q_mean', t.minimum(q1, q2).mean()],
            ['Statistics/q_max', t.maximum(q1, q2).max()]
        ])
