#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from torch import distributions as td

from rls.nn.noised_actions import ClippedNormalNoisedAction
from rls.algos.base.off_policy import Off_Policy
from rls.nn.noised_actions import Noise_action_REGISTER
from rls.nn.models import (CriticQvalueOne,
                           ActorDct,
                           ActorDPG)
from rls.nn.utils import OPLR
from rls.utils.torch_utils import q_target_func
from rls.common.decorator import iTensor_oNumpy


class DPG(Off_Policy):
    '''
    Deterministic Policy Gradient, https://hal.inria.fr/file/index/docid/938992/filename/dpg-icml2014.pdf
    '''
    # off-policy DPG

    def __init__(self,
                 envspec,

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
        super().__init__(envspec=envspec, **kwargs)
        self.discrete_tau = discrete_tau
        self.use_target_action_noise = use_target_action_noise

        if self.is_continuous:
            self.target_noised_action = ClippedNormalNoisedAction(sigma=0.2, noise_bound=0.2)
            self.noised_action = Noise_action_REGISTER[noise_action](**noise_params)
            self.actor = ActorDPG(self.rep_net.h_dim,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_continuous'])
        else:
            self.gumbel_dist = td.gumbel.Gumbel(0, 1)
            self.actor = ActorDct(self.rep_net.h_dim,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_discrete'])

        self.critic = CriticQvalueOne(self.rep_net.h_dim,
                                      action_dim=self.a_dim,
                                      network_settings=network_settings['q'])

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

    def __call__(self, obs, evaluation=False):
        mu, pi = self._get_action(obs)
        return mu if evaluation else pi

    @iTensor_oNumpy
    def _get_action(self, obs):
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
        return mu, pi

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
        feat_, _ = self.rep_net(BATCH.obs_, cell_state=cell_state['obs_'])
        if self.is_continuous:
            action_target = self.actor(feat_)
            if self.use_target_action_noise:
                action_target = self.target_noised_action(action_target)
            mu = output
        else:
            target_logits = self.actor(feat_)
            target_cate_dist = td.categorical.Categorical(logits=target_logits)
            target_pi = target_cate_dist.sample()
            target_log_pi = target_cate_dist.log_prob(target_pi)
            action_target = t.nn.functional.one_hot(target_pi, self.a_dim).float()

            logits = output
            _pi = logits.softmax(-1)
            _pi_true_one_hot = t.nn.functional.one_hot(logits.argmax(-1), self.a_dim).float()
            _pi_diff = (_pi_true_one_hot - _pi).detach()
            mu = _pi_diff + _pi
        q_target = self.critic(feat_, action_target)
        dc_r = q_target_func(BATCH.reward,
                             self.gamma,
                             BATCH.done,
                             q_target)
        q = self.critic(feat, BATCH.action)
        td_error = dc_r - q
        q_loss = 0.5 * (td_error.square() * isw).mean()
        q_actor = self.critic(feat, mu)
        actor_loss = -q_actor.mean()

        self.critic_oplr.step(q_loss)
        self.actor_oplr.step(actor_loss)

        self.global_step.add_(1)
        return td_error, dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/critic_loss', q_loss],
            ['Statistics/q_min', q.min()],
            ['Statistics/q_mean', q.mean()],
            ['Statistics/q_max', q.max()]
        ])
