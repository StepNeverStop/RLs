#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from torch import distributions as td

from rls.algorithms.base.off_policy import Off_Policy
from rls.utils.torch_utils import (squash_action,
                                   tsallis_entropy_log_q,
                                   q_target_func,
                                   sync_params_pairs)
from rls.utils.sundry_utils import LinearAnnealing
from rls.nn.models import (ActorCts,
                           ActorDct,
                           CriticQvalueOne)
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


class TAC(Off_Policy):
    """Tsallis Actor Critic, TAC with V neural Network. https://arxiv.org/abs/1902.00137
    """

    def __init__(self,
                 envspec,

                 alpha=0.2,
                 annealing=True,
                 last_alpha=0.01,
                 ployak=0.995,
                 entropic_index=1.5,
                 discrete_tau=1.0,
                 network_settings={
                     'actor_continuous': {
                         'share': [128, 128],
                         'mu': [64],
                         'log_std': [64],
                         'soft_clip': False,
                         'log_std_bound': [-20, 2]
                     },
                     'actor_discrete': [64, 32],
                     'q': [128, 128]
                 },
                 auto_adaption=True,
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 alpha_lr=5.0e-4,
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.ployak = ployak
        self.discrete_tau = discrete_tau
        self.entropic_index = 2 - entropic_index
        self.auto_adaption = auto_adaption
        self.annealing = annealing

        if self.auto_adaption:
            self.log_alpha = t.tensor(0., requires_grad=True).to(self.device)
        else:
            self.log_alpha = t.tensor(alpha).log().to(self.device)
            if self.annealing:
                self.alpha_annealing = LinearAnnealing(alpha, last_alpha, 1e6)

        self.critic = CriticQvalueOne(self.rep_net.h_dim,
                                      action_dim=self.a_dim, network_settings=network_settings['q']).to(self.device)
        self.critic2 = CriticQvalueOne(self.rep_net.h_dim,
                                       action_dim=self.a_dim, network_settings=network_settings['q']).to(self.device)

        self.critic_target = deepcopy(self.critic)
        self.critic_target.eval()
        self.critic2_target = deepcopy(self.critic2)
        self.critic2_target.eval()
        self._target_rep_net = deepcopy(self.rep_net)
        self._target_rep_net.eval()

        if self.is_continuous:
            self.actor = ActorCts(self.rep_net.h_dim,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_continuous']).to(self.device)
        else:
            self.actor = ActorDct(self.rep_net.h_dim,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_discrete']).to(self.device)

        # entropy = -log(1/|A|) = log |A|
        self.target_entropy = 0.98 * (-self.a_dim if self.is_continuous else np.log(self.a_dim))

        self._pairs = [(self._target_rep_net, self.rep_net),
                       (self.critic_target, self.critic),
                       (self.critic2_target, self.critic2)]
        sync_params_pairs(self._pairs)

        self.actor_oplr = OPLR(self.actor, actor_lr)
        self.critic_oplr = OPLR([self.rep_net, self.critic, self.critic2], critic_lr)
        self.alpha_oplr = OPLR(self.log_alpha, alpha_lr)

        self._worker_modules.update(rep_net=self.rep_net,
                                    actor=self.actor)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(critic=self.critic,
                                     critic2=self.critic2,
                                     log_alpha=self.log_alpha,
                                     actor_oplr=self.actor_oplr,
                                     critic_oplr=self.critic_oplr,
                                     alpha_oplr=self.alpha_oplr)
        self.initialize_data_buffer()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def __call__(self, obs, evaluation=False):
        mu, pi, self.cell_state = self.call(obs, cell_state=self.cell_state)
        return mu if evaluation else pi

    @iTensor_oNumpy
    def call(self, obs, cell_state):
        feat, cell_state = self.rep_net(obs, cell_state=cell_state)
        if self.is_continuous:
            mu, log_std = self.actor(feat)
            pi = td.Normal(mu, log_std.exp()).sample().tanh()
            mu.tanh_()  # squash mu
        else:
            logits = self.actor(feat)
            mu = logits.argmax(1)
            cate_dist = td.Categorical(logits=logits)
            pi = cate_dist.sample()
        return mu, pi, cell_state

    def _target_params_update(self):
        sync_params_pairs(self._pairs, self.ployak)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([
                    ['LEARNING_RATE/actor_lr', self.actor_oplr.lr],
                    ['LEARNING_RATE/critic_lr', self.critic_oplr.lr],
                    ['LEARNING_RATE/alpha_lr', self.alpha_oplr.lr]
                ])
            })

    def _train(self, BATCH, isw, cell_states):
        td_error, summaries = self.train(BATCH, isw, cell_states)
        if self.annealing and not self.auto_adaption:
            self.log_alpha.copy_(self.alpha_annealing(self.global_step).log())
        return td_error, summaries

    @iTensor_oNumpy
    def train(self, BATCH, isw, cell_states):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
        feat_, _ = self._target_rep_net(BATCH.obs_, cell_state=cell_states['obs_'])
        if self.is_continuous:
            target_mu, target_log_std = self.actor(feat_)
            dist = td.Normal(target_mu, target_log_std.exp())
            target_pi = dist.sample()
            target_pi, target_log_pi = squash_action(target_pi, dist.log_prob(target_pi), is_independent=False)
            target_log_pi = tsallis_entropy_log_q(target_log_pi, self.entropic_index)
        else:
            target_logits = self.actor(feat_)
            target_cate_dist = td.Categorical(logits=target_logits)
            target_pi = target_cate_dist.sample()
            target_log_pi = target_cate_dist.log_prob(target_pi)
            target_pi = t.nn.functional.one_hot(target_pi, self.a_dim).float()
        q1 = self.critic(feat, BATCH.action)
        q2 = self.critic2(feat, BATCH.action)

        q1_target = self.critic_target(feat_, target_pi)
        q2_target = self.critic2_target(feat_, target_pi)
        q_target = t.minimum(q1_target, q2_target)
        dc_r = q_target_func(BATCH.reward,
                             self.gamma,
                             BATCH.done,
                             (q_target - self.alpha * target_log_pi))
        td_error1 = q1 - dc_r
        td_error2 = q2 - dc_r
        q1_loss = (td_error1.square() * isw).mean()
        q2_loss = (td_error2.square() * isw).mean()
        critic_loss = 0.5 * q1_loss + 0.5 * q2_loss
        self.critic_oplr.step(critic_loss)

        feat = feat.detach()
        if self.is_continuous:
            mu, log_std = self.actor(feat)
            dist = td.Normal(mu, log_std.exp())
            pi = dist.rsample()
            pi, log_pi = squash_action(pi, dist.log_prob(pi), is_independent=False)
            log_pi = tsallis_entropy_log_q(log_pi, self.entropic_index)
            entropy = dist.entropy().mean()
        else:
            logits = self.actor(feat)
            logp_all = logits.log_softmax(-1)
            gumbel_noise = td.Gumbel(0, 1).sample(BATCH.action.shape)
            _pi = ((logp_all + gumbel_noise) / self.discrete_tau).softmax(-1)
            _pi_true_one_hot = t.nn.functional.one_hot(_pi.argmax(1), self.a_dim).float()
            _pi_diff = (_pi_true_one_hot - _pi).detach()
            pi = _pi_diff + _pi
            log_pi = (logp_all * pi).sum(1, keepdim=True)
            entropy = -(logp_all.exp() * logp_all).sum(1, keepdim=True).mean()
        q_s_pi = t.minimum(self.critic(feat, pi), self.critic2(feat, pi))
        actor_loss = -(q_s_pi - self.alpha * log_pi).mean()
        self.actor_oplr.step(actor_loss)

        if self.auto_adaption:
            alpha_loss = -(self.alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_oplr.step(alpha_loss)

        self.global_step.add_(1)
        summaries = dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/q1_loss', q1_loss],
            ['LOSS/q2_loss', q2_loss],
            ['LOSS/critic_loss', critic_loss],
            ['Statistics/log_alpha', self.log_alpha],
            ['Statistics/alpha', self.alpha],
            ['Statistics/entropy', entropy],
            ['Statistics/q_min', t.minimum(q1, q2).min()],
            ['Statistics/q_mean', t.minimum(q1, q2).mean()],
            ['Statistics/q_max', t.maximum(q1, q2).max()]
        ])
        if self.auto_adaption:
            summaries.update({
                'LOSS/alpha_loss': alpha_loss
            })
        return (td_error1 + td_error2) / 2, summaries
