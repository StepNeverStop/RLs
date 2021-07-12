#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from torch import distributions as td

from rls.utils.torch_utils import (squash_rsample,
                                   gaussian_entropy,
                                   q_target_func,
                                   sync_params_pairs)
from rls.algos.base.off_policy import Off_Policy
from rls.utils.sundry_utils import LinearAnnealing
from rls.nn.models import (ActorDct,
                           ActorCts,
                           CriticQvalueOne,
                           CriticQvalueAll)
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


class SAC(Off_Policy):
    """
        Soft Actor-Critic Algorithms and Applications. https://arxiv.org/abs/1812.05905
        Soft Actor-Critic for Discrete Action Settings. https://arxiv.org/abs/1910.07207
    """

    def __init__(self,
                 envspec,

                 alpha=0.2,
                 annealing=True,
                 last_alpha=0.01,
                 ployak=0.995,
                 use_gumbel=True,
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
        self.use_gumbel = use_gumbel
        self.discrete_tau = discrete_tau
        self.auto_adaption = auto_adaption
        self.annealing = annealing

        if self.auto_adaption:
            self.log_alpha = t.tensor(0., requires_grad=True)
        else:
            self.log_alpha = t.tensor(alpha).log()
            if self.annealing:
                self.alpha_annealing = LinearAnnealing(alpha, last_alpha, 1e6)

        # entropy = -log(1/|A|) = log |A|
        self.target_entropy = 0.98 * (-self.a_dim if self.is_continuous else np.log(self.a_dim))

        if self.is_continuous or self.use_gumbel:
            self.critic = CriticQvalueOne(self.rep_net.h_dim,
                                          action_dim=self.a_dim,
                                          network_settings=network_settings['q']).to(self.device)
            self.critic2 = CriticQvalueOne(self.rep_net.h_dim,
                                           action_dim=self.a_dim,
                                           network_settings=network_settings['q']).to(self.device)
        else:
            self.critic = CriticQvalueAll(self.rep_net.h_dim,
                                          output_shape=self.a_dim,
                                          network_settings=network_settings['q']).to(self.device)
            self.critic2 = CriticQvalueAll(self.rep_net.h_dim,
                                           output_shape=self.a_dim,
                                           network_settings=network_settings['q']).to(self.device)

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
            if self.use_gumbel:
                self.gumbel_dist = td.gumbel.Gumbel(0, 1)

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

    @iTensor_oNumpy
    def __call__(self, obs, evaluation=False):
        feat, self.cell_state = self.rep_net(obs, cell_state=self.cell_state)
        if self.is_continuous:
            mu, log_std = self.actor(feat)
            pi, _ = squash_rsample(mu, log_std)
            mu.tanh_()  # squash mu
        else:
            logits = self.actor(feat)
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
                    ['LEARNING_RATE/critic_lr', self.critic_oplr.lr],
                    ['LEARNING_RATE/alpha_lr', self.alpha_oplr.lr]
                ])
            })

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _train(self, BATCH, isw, cell_states):
        if self.is_continuous or self.use_gumbel:
            td_error, summaries = self.train_continuous(BATCH, isw, cell_states)
        else:
            td_error, summaries = self.train_discrete(BATCH, isw, cell_states)
        if self.annealing and not self.auto_adaption:
            self.log_alpha.copy_(self.alpha_annealing(self.global_step).log())
        return td_error, summaries

    @iTensor_oNumpy
    def train_continuous(self, BATCH, isw, cell_states):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
        feat_, _ = self._target_rep_net(BATCH.obs_, cell_state=cell_states['obs_'])
        if self.is_continuous:
            mu, log_std = self.actor(feat)
            pi, log_pi = squash_rsample(mu, log_std)
            entropy = gaussian_entropy(log_std)
            target_mu, target_log_std = self.actor(feat_)
            target_pi, target_log_pi = squash_rsample(target_mu, target_log_std)
        else:
            logits = self.actor(feat)
            logp_all = logits.log_softmax(-1)
            gumbel_noise = self.gumbel_dist.sample(BATCH.action.shape)
            _pi = ((logp_all + gumbel_noise) / self.discrete_tau).softmax(-1)
            _pi_true_one_hot = t.nn.functional.one_hot(_pi.argmax(-1), self.a_dim).float()
            _pi_diff = (_pi_true_one_hot - _pi).detach()
            pi = _pi_diff + _pi
            log_pi = (logp_all * pi).sum(1, keepdim=True)
            entropy = -(logp_all.exp() * logp_all).sum(1, keepdim=True).mean()

            target_logits = self.actor(feat_)
            target_cate_dist = td.categorical.Categorical(logits=target_logits)
            target_pi = target_cate_dist.sample()
            target_log_pi = target_cate_dist.log_prob(target_pi)
            target_pi = t.nn.functional.one_hot(target_pi, self.a_dim).float()
        q1 = self.critic(feat, BATCH.action)
        q2 = self.critic2(feat, BATCH.action)
        q_s_pi = t.minimum(self.critic(feat, pi), self.critic2(feat, pi))
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
        actor_loss = -(q_s_pi - self.alpha * log_pi).mean()

        self.critic_oplr.step(critic_loss)
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

    @iTensor_oNumpy
    def train_discrete(self, BATCH, isw, cell_states):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
        feat_, _ = self._target_rep_net(BATCH.obs_, cell_state=cell_states['obs_'])
        q1_all = self.critic(feat)
        q2_all = self.critic2(feat)  # [B, A]

        logits = self.actor(feat)
        logp_all = logits.log_softmax(-1)
        entropy = -(logp_all.exp() * logp_all).sum(1, keepdim=True)    # [B, 1]
        q_all = t.minimum(q1_all, q2_all)  # [B, A]

        def q_function(x): return (x * BATCH.action).sum(-1, keepdim=True)  # [B, 1]
        q1 = q_function(q1_all)
        q2 = q_function(q2_all)
        target_logits = self.actor(feat_)  # [B, A]
        target_log_probs = target_logits.log_softmax(-1)  # [B, A]
        q1_target = self.critic_target(feat_)
        q1_target = self.critic2_target(feat_)  # [B, A]

        def v_target_function(x): return (target_log_probs.exp() * (x - self.alpha * target_log_probs)).sum(-1, keepdim=True)  # [B, 1]
        v1_target = v_target_function(q1_target)
        v2_target = v_target_function(q2_target)
        v_target = t.minimum(v1_target, v2_target)
        dc_r = q_target_func(BATCH.reward,
                             self.gamma,
                             BATCH.done,
                             v_target)
        td_error1 = q1 - dc_r
        td_error2 = q2 - dc_r
        q1_loss = (td_error1.square() * isw).mean()
        q2_loss = (td_error2.square() * isw).mean()

        critic_loss = 0.5 * q1_loss + 0.5 * q2_loss
        actor_loss = -((q_all - self.alpha * logp_all) * logp_all.exp()).sum().mean()  # [B, A] => [B,]
        # actor_loss = - (q_all + self.alpha * entropy).mean()

        self.critic_oplr.step(critic_loss)
        self.actor_oplr.step(actor_loss)
        if self.auto_adaption:
            corr = (self.target_entropy - entropy).detach()
            # corr = ((logp_all - self.a_dim) * logp_all.exp()).sum(-1).detach()    #[B, A] => [B,]
            # J(\alpha)=\pi_{t}\left(s_{t}\right)^{T}\left[-\alpha\left(\log \left(\pi_{t}\left(s_{t}\right)\right)+\bar{H}\right)\right]
            # \bar{H} is negative
            alpha_loss = -(self.alpha * corr).mean()
            self.alpha_oplr.step(alpha_loss)

        self.global_step.add_(1)
        summaries = dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/q1_loss', q1_loss],
            ['LOSS/q2_loss', q2_loss],
            ['LOSS/critic_loss', critic_loss],
            ['Statistics/log_alpha', self.log_alpha],
            ['Statistics/alpha', self.alpha],
            ['Statistics/entropy', entropy.mean()]
        ])
        if self.auto_adaption:
            summaries.update({
                'LOSS/alpha_loss': alpha_loss
            })
        return (td_error1 + td_error2) / 2, summaries
