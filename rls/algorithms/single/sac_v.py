#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from torch import distributions as td

from rls.algorithms.base.off_policy import Off_Policy
from rls.utils.torch_utils import (squash_action,
                                   q_target_func,
                                   sync_params_list)
from rls.utils.sundry_utils import LinearAnnealing
from rls.nn.models import (CriticValue,
                           ActorDct,
                           ActorCts,
                           CriticQvalueOne,
                           CriticQvalueAll)
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


class SAC_V(Off_Policy):
    """
        Soft Actor Critic with Value neural network. https://arxiv.org/abs/1812.05905
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
                     'q': [128, 128],
                     'v': [128, 128]
                 },
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 alpha_lr=5.0e-4,
                 auto_adaption=True,
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.ployak = ployak
        self.use_gumbel = use_gumbel
        self.discrete_tau = discrete_tau
        self.auto_adaption = auto_adaption
        self.annealing = annealing

        if self.auto_adaption:
            self.log_alpha = t.tensor(0., requires_grad=True).to(self.device)
        else:
            self.log_alpha = t.tensor(alpha).log().to(self.device)
            if self.annealing:
                self.alpha_annealing = LinearAnnealing(alpha, last_alpha, 1e6)

        self.v_net = CriticValue(self.rep_net.h_dim,
                                 network_settings=network_settings['v']).to(self.device)
        self.v_target_net = deepcopy(self.v_net)
        self.v_target_net.eval()
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

        if self.is_continuous or self.use_gumbel:
            self.q_net = CriticQvalueOne(self.rep_net.h_dim,
                                         action_dim=self.a_dim,
                                         network_settings=network_settings['q']).to(self.device)
            self.q_net2 = CriticQvalueOne(self.rep_net.h_dim,
                                          action_dim=self.a_dim,
                                          network_settings=network_settings['q']).to(self.device)
        else:
            self.q_net = CriticQvalueAll(self.rep_net.h_dim,
                                         action_dim=self.a_dim,
                                         network_settings=network_settings['q']).to(self.device)
            self.q_net2 = CriticQvalueAll(self.rep_net.h_dim,
                                          action_dim=self.a_dim,
                                          network_settings=network_settings['q']).to(self.device)

        self._pairs = [(self.v_target_net, self._target_rep_net),
                       (self.v_net, self.rep_net)]
        sync_params_list(self._pairs)

        self.actor_oplr = OPLR(self.actor, actor_lr)
        self.critic_oplr = OPLR([self.rep_net, self.q_net, self.q_net2, self.v_net], critic_lr)
        self.alpha_oplr = OPLR(self.log_alpha, alpha_lr)

        self._worker_modules.update(rep_net=self.rep_net,
                                    actor=self.actor)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(v_net=self.v_net,
                                     q_net=self.q_net,
                                     q_net2=self.q_net2,
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
            mu.tanh_()    # squash mu
        else:
            logits = self.actor(feat)
            mu = logits.argmax(1)
            cate_dist = td.Categorical(logits=logits)
            pi = cate_dist.sample()
        return mu, pi, cell_state

    def _target_params_update(self):
        sync_params_list(self._pairs, self.ployak)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([
                    ['LEARNING_RATE/actor_lr', self.actor_oplr.lr],
                    ['LEARNING_RATE/critic_lr', self.critic_oplr.lr],
                    ['LEARNING_RATE/alpha_lr', self.alpha_oplr.lr]
                ]),
            })

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
        v = self.v_net(feat)
        v_target = self.v_target_net(feat_)

        if self.is_continuous:
            mu, log_std = self.actor(feat)
            dist = td.Normal(mu, log_std.exp())
            pi = dist.rsample()
            pi, log_pi = squash_action(pi, dist.log_prob(pi))
        else:
            logits = self.actor(feat)
            logp_all = logits.log_softmax(-1)
            gumbel_noise = td.Gumbel(0, 1).sample(BATCH.action.shape)
            _pi = ((logp_all + gumbel_noise) / self.discrete_tau).softmax(-1)
            _pi_true_one_hot = t.nn.functional.one_hot(_pi.argmax(-1), self.a_dim).float()
            _pi_diff = (_pi_true_one_hot - _pi).detach()
            pi = _pi_diff + _pi
            log_pi = (logp_all * pi).sum(1, keepdim=True)
        q1 = self.q_net(feat, BATCH.action)
        q2 = self.q_net2(feat, BATCH.action)
        q1_pi = self.q_net(feat, pi)
        q2_pi = self.q_net2(feat, pi)
        dc_r = q_target_func(BATCH.reward,
                             self.gamma,
                             BATCH.done,
                             v_target)
        v_from_q_stop = (t.minimum(q1_pi, q2_pi) - self.alpha * log_pi).detach()
        td_v = v - v_from_q_stop
        td_error1 = q1 - dc_r
        td_error2 = q2 - dc_r
        q1_loss = (td_error1.square() * isw).mean()
        q2_loss = (td_error2.square() * isw).mean()
        v_loss_stop = (td_v.square() * isw).mean()
        critic_loss = 0.5 * q1_loss + 0.5 * q2_loss + 0.5 * v_loss_stop
        self.critic_oplr.step(critic_loss)

        feat = feat.detach()
        if self.is_continuous:
            mu, log_std = self.actor(feat)
            dist = td.Normal(mu, log_std.exp())
            pi = dist.rsample()
            pi, log_pi = squash_action(pi, dist.log_prob(pi))
            entropy = dist.entropy().mean()
        else:
            logits = self.actor(feat)
            logp_all = logits.log_softmax(-1)
            gumbel_noise = td.Gumbel(0, 1).sample(BATCH.action.shape)
            _pi = ((logp_all + gumbel_noise) / self.discrete_tau).softmax(-1)
            _pi_true_one_hot = t.nn.functional.one_hot(_pi.argmax(-1), self.a_dim).float()
            _pi_diff = (_pi_true_one_hot - _pi).detach()
            pi = _pi_diff + _pi
            log_pi = (logp_all * pi).sum(1, keepdim=True)
            entropy = -(logp_all.exp() * logp_all).sum(1, keepdim=True).mean()
        q1_pi = self.q_net(feat, pi)
        actor_loss = -(q1_pi - self.alpha * log_pi).mean()
        self.actor_oplr.step(actor_loss)

        if self.auto_adaption:
            alpha_loss = -(self.alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_oplr.step(alpha_loss)

        self.global_step.add_(1)
        summaries = dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/q1_loss', q1_loss],
            ['LOSS/q2_loss', q2_loss],
            ['LOSS/v_loss', v_loss_stop],
            ['LOSS/critic_loss', critic_loss],
            ['Statistics/log_alpha', self.log_alpha],
            ['Statistics/alpha', self.alpha],
            ['Statistics/entropy', entropy],
            ['Statistics/q_min', t.minimum(q1, q2).min()],
            ['Statistics/q_mean', t.minimum(q1, q2).mean()],
            ['Statistics/q_max', t.maximum(q1, q2).max()],
            ['Statistics/v_mean', v.mean()]
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
        v = self.v_net(feat)  # [B, 1]
        v_target = self.v_target_net(feat_)  # [B, 1]

        q1_all = self.q_net(feat)
        q2_all = self.q_net2(feat)   # [B, A]
        def q_function(x): return (x * BATCH.action).sum(-1, keepdim=True)  # [B, 1]
        q1 = q_function(q1_all)
        q2 = q_function(q2_all)
        logits = self.actor(feat)  # [B, A]
        logp_all = logits.log_softmax(-1)  # [B, A]

        q_all = t.minimum(self.q_net(feat), self.q_net2(feat))   # [B, A]
        actor_loss = -((q_all - self.alpha * logp_all) * logp_all.exp()).sum().mean()  # [B, A] => [B,]

        dc_r = q_target_func(BATCH.reward,
                             self.gamma,
                             BATCH.done,
                             v_target)
        td_v = v - (t.minimum(
            (logp_all.exp() * q1_all).sum(-1),
            (logp_all.exp() * q2_all).sum(-1)
        )).detach()
        td_error1 = q1 - dc_r
        td_error2 = q2 - dc_r
        q1_loss = (td_error1.square() * isw).mean()
        q2_loss = (td_error2.square() * isw).mean()
        v_loss_stop = (td_v.square() * isw).mean()
        critic_loss = 0.5 * q1_loss + 0.5 * q2_loss + 0.5 * v_loss_stop
        self.critic_oplr.step(critic_loss)

        feat = feat.detach()
        q1_all = self.q_net(feat)
        q2_all = self.q_net2(feat)   # [B, A]
        logits = self.actor(feat)  # [B, A]
        logp_all = logits.log_softmax(-1)  # [B, A]

        entropy = -(logp_all.exp() * logp_all).sum(1, keepdim=True)    # [B, 1]
        q_all = t.minimum(self.q_net(feat), self.q_net2(feat))   # [B, A]
        actor_loss = -((q_all - self.alpha * logp_all) * logp_all.exp()).sum().mean()  # [B, A] => [B,]
        self.actor_oplr.step(actor_loss)

        if self.auto_adaption:
            corr = (self.target_entropy - entropy).detach()
            # corr = ((logp_all - self.a_dim) * logp_all.exp()).sum(-1).detach()    #[B, A] => [B,]
            alpha_loss = -(self.alpha * corr).mean()
            self.alpha_oplr.step(alpha_loss)

        self.global_step.add_(1)
        summaries = dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/q1_loss', q1_loss],
            ['LOSS/q2_loss', q2_loss],
            ['LOSS/v_loss', v_loss_stop],
            ['LOSS/critic_loss', critic_loss],
            ['Statistics/log_alpha', self.log_alpha],
            ['Statistics/alpha', self.alpha],
            ['Statistics/entropy', entropy.mean()],
            ['Statistics/v_mean', v.mean()]
        ])
        if self.auto_adaption:
            summaries.update({
                'LOSS/alpha_loss': alpha_loss
            })
        return (td_error1 + td_error2) / 2, summaries
