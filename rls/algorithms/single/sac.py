#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from torch import distributions as td

from rls.utils.torch_utils import (squash_action,
                                   q_target_func)
from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.utils.sundry_utils import LinearAnnealing
from rls.nn.models import (ActorDct,
                           ActorCts,
                           CriticQvalueOne,
                           CriticQvalueAll)
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy
from rls.nn.modules.wrappers import TargetTwin
from rls.common.specs import Data


class SAC(SarlOffPolicy):
    """
        Soft Actor-Critic Algorithms and Applications. https://arxiv.org/abs/1812.05905
        Soft Actor-Critic for Discrete Action Settings. https://arxiv.org/abs/1910.07207
    """
    policy_mode = 'off-policy'

    def __init__(self,
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
        super().__init__(**kwargs)
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

        # entropy = -log(1/|A|) = log |A|
        self.target_entropy = 0.98 * \
            (-self.a_dim if self.is_continuous else np.log(self.a_dim))

        if self.is_continuous or self.use_gumbel:
            critic = CriticQvalueOne(self.obs_spec,
                                     rep_net_params=self.rep_net_params,
                                     action_dim=self.a_dim,
                                     network_settings=network_settings['q'])
        else:
            critic = CriticQvalueAll(self.obs_spec,
                                     rep_net_params=self.rep_net_params,
                                     output_shape=self.a_dim,
                                     network_settings=network_settings['q'])
        self.critic = TargetTwin(critic, self.ployak).to(self.device)
        self.critic2 = deepcopy(self.critic)

        if self.is_continuous:
            self.actor = ActorCts(self.obs_spec,
                                  rep_net_params=self.rep_net_params,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_continuous']).to(self.device)
        else:
            self.actor = ActorDct(self.obs_spec,
                                  rep_net_params=self.rep_net_params,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_discrete']).to(self.device)

        self.actor_oplr = OPLR(self.actor, actor_lr)
        self.critic_oplr = OPLR([self.critic, self.critic2], critic_lr)
        self.alpha_oplr = OPLR(self.log_alpha, alpha_lr)

        self._trainer_modules.update(actor=self.actor,
                                     critic=self.critic,
                                     critic2=self.critic2,
                                     log_alpha=self.log_alpha,
                                     actor_oplr=self.actor_oplr,
                                     critic_oplr=self.critic_oplr,
                                     alpha_oplr=self.alpha_oplr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @iTensor_oNumpy
    def select_action(self, obs):
        if self.is_continuous:
            mu, log_std = self.actor(
                obs, cell_state=self.cell_state)   # [B, A]
            pi = td.Normal(mu, log_std.exp()).sample().tanh()
            mu.tanh_()  # squash mu  # [B, A]
        else:
            logits = self.actor(obs, cell_state=self.cell_state)     # [B, A]
            mu = logits.argmax(-1)    # [B,]
            cate_dist = td.Categorical(logits=logits)
            pi = cate_dist.sample()  # [B,]
        self.next_cell_state = self.actor.get_cell_state()
        actions = pi if self._is_train_mode else mu
        return actions, Data(action=actions)

    def _train(self, BATCH):
        if self.is_continuous or self.use_gumbel:
            td_error, summaries = self._train_continuous(BATCH)
        else:
            td_error, summaries = self._train_discrete(BATCH)
        return td_error, summaries

    @iTensor_oNumpy
    def _train_continuous(self, BATCH):
        q1 = self.critic(BATCH.obs, BATCH.action)   # [T, B, 1]
        q2 = self.critic2(BATCH.obs, BATCH.action)   # [T, B, 1]
        if self.is_continuous:
            target_mu, target_log_std = self.actor(BATCH.obs_)   # [T, B, A]
            dist = td.Normal(target_mu, target_log_std.exp())
            target_pi = dist.sample()   # [T, B, A]
            target_pi, target_log_pi = squash_action(
                target_pi, dist.log_prob(target_pi))   # [T, B, A], [T, B, 1]
        else:
            target_logits = self.actor(BATCH.obs_)  # [T, B, A]
            target_cate_dist = td.Categorical(logits=target_logits)
            target_pi = target_cate_dist.sample()   # [T, B, A]
            target_log_pi = target_cate_dist.log_prob(
                target_pi).unsqueeze(-1)  # [T, B, 1]
            target_pi = t.nn.functional.one_hot(
                target_pi, self.a_dim).float()  # [T, B, A]
        q1_target = self.critic.t(BATCH.obs_, target_pi)    # [T, B, 1]
        q2_target = self.critic2.t(BATCH.obs_, target_pi)   # [T, B, 1]
        q_target = t.minimum(q1_target, q2_target)  # [T, B, 1]
        dc_r = q_target_func(BATCH.reward,
                             self.gamma,
                             BATCH.done,
                             (q_target - self.alpha * target_log_pi),
                             BATCH.begin_mask,
                             use_rnn=self.use_rnn)  # [T, B, 1]
        td_error1 = q1 - dc_r   # [T, B, 1]
        td_error2 = q2 - dc_r   # [T, B, 1]
        q1_loss = (td_error1.square() * BATCH.get('isw', 1.0)).mean()    # 1
        q2_loss = (td_error2.square() * BATCH.get('isw', 1.0)).mean()    # 1
        critic_loss = 0.5 * q1_loss + 0.5 * q2_loss
        self.critic_oplr.step(critic_loss)

        if self.is_continuous:
            mu, log_std = self.actor(BATCH.obs)  # [T, B, A]
            dist = td.Normal(mu, log_std.exp())
            pi = dist.rsample()  # [T, B, A]
            pi, log_pi = squash_action(
                pi, dist.log_prob(pi))   # [T, B, A], [T, B, 1]
            entropy = dist.entropy().mean()  # 1
        else:
            logits = self.actor(BATCH.obs)  # [T, B, A]
            logp_all = logits.log_softmax(-1)   # [T, B, A]
            gumbel_noise = td.Gumbel(0, 1).sample(logp_all.shape)   # [T, B, A]
            _pi = ((logp_all + gumbel_noise) /
                   self.discrete_tau).softmax(-1)   # [T, B, A]
            _pi_true_one_hot = t.nn.functional.one_hot(
                _pi.argmax(-1), self.a_dim).float()  # [T, B, A]
            _pi_diff = (_pi_true_one_hot - _pi).detach()    # [T, B, A]
            pi = _pi_diff + _pi  # [T, B, A]
            log_pi = (logp_all * pi).sum(-1, keepdim=True)   # [T, B, 1]
            entropy = -(logp_all.exp() * logp_all).sum(-1).mean()   # 1
        q_s_pi = t.minimum(self.critic(BATCH.obs, pi),
                           self.critic2(BATCH.obs, pi))  # [T, B, 1]

        actor_loss = -(q_s_pi - self.alpha * log_pi).mean()  # 1

        self.actor_oplr.step(actor_loss)
        if self.auto_adaption:
            alpha_loss = - \
                (self.alpha * (log_pi + self.target_entropy).detach()).mean()  # 1
            self.alpha_oplr.step(alpha_loss)

        summaries = dict([
            ['LEARNING_RATE/actor_lr', self.actor_oplr.lr],
            ['LEARNING_RATE/critic_lr', self.critic_oplr.lr],
            ['LEARNING_RATE/alpha_lr', self.alpha_oplr.lr],
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
    def _train_discrete(self, BATCH):
        q1_all = self.critic(BATCH.obs)  # [T, B, A]
        q2_all = self.critic2(BATCH.obs)  # [T, B, A]

        q1 = (q1_all * BATCH.action).sum(-1, keepdim=True)  # [T, B, 1]
        q2 = (q2_all * BATCH.action).sum(-1, keepdim=True)  # [T, B, 1]
        target_logits = self.actor(BATCH.obs_)  # [T, B, A]
        target_log_probs = target_logits.log_softmax(-1)  # [T, B, A]
        q1_target = self.critic.t(BATCH.obs_)   # [T, B, A]
        q2_target = self.critic2.t(BATCH.obs_)  # [T, B, A]

        def v_target_function(x): return (target_log_probs.exp(
        ) * (x - self.alpha * target_log_probs)).sum(-1, keepdim=True)  # [T, B, 1]
        v1_target = v_target_function(q1_target)    # [T, B, 1]
        v2_target = v_target_function(q2_target)    # [T, B, 1]
        v_target = t.minimum(v1_target, v2_target)   # [T, B, 1]
        dc_r = q_target_func(BATCH.reward,
                             self.gamma,
                             BATCH.done,
                             v_target,
                             BATCH.begin_mask,
                             use_rnn=self.use_rnn)  # [T, B, 1]
        td_error1 = q1 - dc_r  # [T, B, 1]
        td_error2 = q2 - dc_r  # [T, B, 1]

        q1_loss = (td_error1.square() * BATCH.get('isw', 1.0)).mean()    # 1
        q2_loss = (td_error2.square() * BATCH.get('isw', 1.0)).mean()    # 1
        critic_loss = 0.5 * q1_loss + 0.5 * q2_loss
        self.critic_oplr.step(critic_loss)

        q1_all = self.critic(BATCH.obs)  # [T, B, A]
        q2_all = self.critic2(BATCH.obs)  # [T, B, A]

        logits = self.actor(BATCH.obs)  # [T, B, A]
        logp_all = logits.log_softmax(-1)   # [T, B, A]
        entropy = -(logp_all.exp() * logp_all).sum(-1,
                                                   keepdim=True)    # [T, B, 1]
        q_all = t.minimum(q1_all, q2_all)  # [T, B, A]
        actor_loss = -((q_all - self.alpha * logp_all) *
                       logp_all.exp()).sum(-1)  # [T, B, A] => [T, B]
        actor_loss = actor_loss.mean()  # 1
        # actor_loss = - (q_all + self.alpha * entropy).mean()

        self.actor_oplr.step(actor_loss)
        if self.auto_adaption:
            corr = (self.target_entropy - entropy).detach()  # [T, B, 1]
            # corr = ((logp_all - self.a_dim) * logp_all.exp()).sum(-1).detach()    #[B, A] => [B,]
            # J(\alpha)=\pi_{t}\left(s_{t}\right)^{T}\left[-\alpha\left(\log \left(\pi_{t}\left(s_{t}\right)\right)+\bar{H}\right)\right]
            # \bar{H} is negative
            alpha_loss = -(self.alpha * corr)    # [T, B, 1]
            alpha_loss = alpha_loss.mean()  # 1
            self.alpha_oplr.step(alpha_loss)

        summaries = dict([
            ['LEARNING_RATE/actor_lr', self.actor_oplr.lr],
            ['LEARNING_RATE/critic_lr', self.critic_oplr.lr],
            ['LEARNING_RATE/alpha_lr', self.alpha_oplr.lr],
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

    def _after_train(self):
        super()._after_train()
        if self.annealing and not self.auto_adaption:
            self.log_alpha.copy_(
                self.alpha_annealing(self.cur_train_step).log())
        self.critic.sync()
        self.critic2.sync()
