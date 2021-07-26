#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from torch import distributions as td

from rls.nn.noised_actions import ClippedNormalNoisedAction
from rls.algos.base.off_policy import Off_Policy
from rls.utils.torch_utils import (sync_params_pairs,
                                   q_target_func)
from rls.nn.noised_actions import Noise_action_REGISTER
from rls.nn.models import (CriticQvalueOne,
                           ActorDct,
                           ActorDPG)
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


class DDPG(Off_Policy):
    '''
    Deep Deterministic Policy Gradient, https://arxiv.org/abs/1509.02971
    '''

    def __init__(self,
                 envspec,

                 ployak=0.995,
                 noise_action='ou',
                 noise_params={
                     'sigma': 0.2
                 },
                 use_target_action_noise=False,
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
        self.discrete_tau = discrete_tau
        self.use_target_action_noise = use_target_action_noise

        if self.is_continuous:
            self.actor = ActorDPG(self.rep_net.h_dim,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_continuous'])
            self.target_noised_action = ClippedNormalNoisedAction(sigma=0.2, noise_bound=0.2)
            if noise_action == 'ou':
                self.noised_action = Noise_action_REGISTER[noise_action](**noise_params)
            elif noise_action == 'normal':
                self.noised_action = self.target_noised_action
            else:
                raise Exception(f'cannot use noised action type of {noise_action}')
        else:
            self.actor = ActorDct(self.rep_net.h_dim,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_discrete']).to(self.device)
            self.gumbel_dist = td.gumbel.Gumbel(0, 1)
        self.critic = CriticQvalueOne(self.rep_net.h_dim,
                                      action_dim=self.a_dim,
                                      network_settings=network_settings['q']).to(self.device)

        self._target_rep_net = deepcopy(self.rep_net)
        self._target_rep_net.eval()
        self.actor_target = deepcopy(self.actor)
        self.actor_target.eval()
        self.critic_target = deepcopy(self.critic)
        self.critic_target.eval()

        self._pairs = [(self._target_rep_net, self.rep_net),
                       (self.actor_target, self.actor),
                       (self.critic_target, self.critic)]
        sync_params_pairs(self._pairs)

        self.actor_oplr = OPLR(self.actor, actor_lr)
        self.critic_oplr = OPLR([self.critic, self.rep_net], critic_lr)

        self._worker_modules.update(rep_net=self.rep_net,
                                    actor=self.actor)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(critic=self.critic,
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
            cate_dist = td.categorical.Categorical(logits=logits)
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
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
        feat_, _ = self._target_rep_net(BATCH.obs_, cell_state=cell_states['obs_'])

        if self.is_continuous:
            action_target = self.actor_target(feat_)
            if self.use_target_action_noise:
                action_target = self.target_noised_action(action_target)
        else:
            target_logits = self.actor_target(feat_)
            target_cate_dist = td.categorical.Categorical(logits=target_logits)
            target_pi = target_cate_dist.sample()
            target_log_pi = target_cate_dist.log_prob(target_pi)
            action_target = t.nn.functional.one_hot(target_pi, self.a_dim).float()
        q = self.critic(feat, BATCH.action)
        q_target = self.critic_target(feat_, action_target)
        dc_r = q_target_func(BATCH.reward,
                             self.gamma,
                             BATCH.done,
                             q_target)
        td_error = dc_r - q
        q_loss = 0.5 * (td_error.square() * isw).mean()
        self.critic_oplr.step(q_loss)

        feat = feat.detach()
        if self.is_continuous:
            mu = self.actor(feat)
        else:
            gumbel_noise = self.gumbel_dist.sample(BATCH.action.shape)
            logits = self.actor(feat)
            logp_all = logits.log_softmax(-1)
            _pi = ((logp_all + gumbel_noise) / self.discrete_tau).softmax(-1)
            _pi_true_one_hot = t.nn.functional.one_hot(_pi.argmax(-1), self.a_dim).float()
            _pi_diff = (_pi_true_one_hot - _pi).detach()
            mu = _pi_diff + _pi
        q_actor = self.critic(feat, mu)
        actor_loss = -q_actor.mean()
        self.actor_oplr.step(actor_loss)

        self.global_step.add_(1)
        return td_error, dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/critic_loss', q_loss],
            ['Statistics/q_min', q.min()],
            ['Statistics/q_mean', q.mean()],
            ['Statistics/q_max', q.max()]
        ])
