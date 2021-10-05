#!/usr/bin/env python3
# encoding: utf-8

from copy import deepcopy

import numpy as np
import torch as th
import torch.distributions as td
import torch.nn.functional as F

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.common.data import Data
from rls.common.decorator import iton
from rls.nn.models import (ActorCts, ActorDct, CriticQvalueAll,
                           CriticQvalueOne, CriticValue)
from rls.nn.modules.wrappers import TargetTwin
from rls.nn.utils import OPLR
from rls.utils.sundry_utils import LinearAnnealing
from rls.utils.torch_utils import n_step_return, squash_action


class SAC_V(SarlOffPolicy):
    """
        Soft Actor Critic with Value neural network. https://arxiv.org/abs/1812.05905
        Soft Actor-Critic for Discrete Action Settings. https://arxiv.org/abs/1910.07207
    """
    policy_mode = 'off-policy'

    def __init__(self,
                 alpha=0.2,
                 annealing=True,
                 last_alpha=0.01,
                 polyak=0.995,
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
        super().__init__(**kwargs)
        self.polyak = polyak
        self.use_gumbel = use_gumbel
        self.discrete_tau = discrete_tau
        self.auto_adaption = auto_adaption
        self.annealing = annealing

        self.v_net = TargetTwin(CriticValue(self.obs_spec,
                                            rep_net_params=self._rep_net_params,
                                            network_settings=network_settings['v']),
                                self.polyak).to(self.device)

        if self.is_continuous:
            self.actor = ActorCts(self.obs_spec,
                                  rep_net_params=self._rep_net_params,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_continuous']).to(self.device)
        else:
            self.actor = ActorDct(self.obs_spec,
                                  rep_net_params=self._rep_net_params,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_discrete']).to(self.device)

        # entropy = -log(1/|A|) = log |A|
        self.target_entropy = 0.98 * (-self.a_dim if self.is_continuous else np.log(self.a_dim))

        if self.is_continuous or self.use_gumbel:
            self.q_net = CriticQvalueOne(self.obs_spec,
                                         rep_net_params=self._rep_net_params,
                                         action_dim=self.a_dim,
                                         network_settings=network_settings['q']).to(self.device)
        else:
            self.q_net = CriticQvalueAll(self.obs_spec,
                                         rep_net_params=self._rep_net_params,
                                         output_shape=self.a_dim,
                                         network_settings=network_settings['q']).to(self.device)
        self.q_net2 = deepcopy(self.q_net)

        self.actor_oplr = OPLR(self.actor, actor_lr, **self._oplr_params)
        self.critic_oplr = OPLR([self.q_net, self.q_net2, self.v_net], critic_lr, **self._oplr_params)

        if self.auto_adaption:
            self.log_alpha = th.tensor(0., requires_grad=True).to(self.device)
            self.alpha_oplr = OPLR(self.log_alpha, alpha_lr, **self._oplr_params)
            self._trainer_modules.update(alpha_oplr=self.alpha_oplr)
        else:
            self.log_alpha = th.tensor(alpha).log().to(self.device)
            if self.annealing:
                self.alpha_annealing = LinearAnnealing(alpha, last_alpha, int(1e6))

        self._trainer_modules.update(actor=self.actor,
                                     v_net=self.v_net,
                                     q_net=self.q_net,
                                     q_net2=self.q_net2,
                                     log_alpha=self.log_alpha,
                                     actor_oplr=self.actor_oplr,
                                     critic_oplr=self.critic_oplr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @iton
    def select_action(self, obs):
        if self.is_continuous:
            mu, log_std = self.actor(obs, rnncs=self.rnncs)  # [B, A]
            pi = td.Normal(mu, log_std.exp()).sample().tanh()  # [B, A]
            mu.tanh_()  # squash mu   # [B, A]
        else:
            logits = self.actor(obs, rnncs=self.rnncs)  # [B, A]
            mu = logits.argmax(-1)  # [B,]
            cate_dist = td.Categorical(logits=logits)
            pi = cate_dist.sample()  # [B,]
        self.rnncs_ = self.actor.get_rnncs()
        actions = pi if self._is_train_mode else mu
        return actions, Data(action=actions)

    def _train(self, BATCH):
        if self.is_continuous or self.use_gumbel:
            td_error = self._train_continuous(BATCH)
        else:
            td_error = self._train_discrete(BATCH)
        return td_error

    @iton
    def _train_continuous(self, BATCH):
        v = self.v_net(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, 1]
        v_target = self.v_net.t(BATCH.obs_, begin_mask=BATCH.begin_mask)  # [T, B, 1]

        if self.is_continuous:
            mu, log_std = self.actor(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            pi = dist.rsample()  # [T, B, A]
            pi, log_pi = squash_action(pi, dist.log_prob(pi).unsqueeze(-1))  # [T, B, A], [T, B, 1]
        else:
            logits = self.actor(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
            logp_all = logits.log_softmax(-1)  # [T, B, A]
            gumbel_noise = td.Gumbel(0, 1).sample(logp_all.shape)  # [T, B, A]
            _pi = ((logp_all + gumbel_noise) / self.discrete_tau).softmax(-1)  # [T, B, A]
            _pi_true_one_hot = F.one_hot(_pi.argmax(-1), self.a_dim).float()  # [T, B, A]
            _pi_diff = (_pi_true_one_hot - _pi).detach()  # [T, B, A]
            pi = _pi_diff + _pi  # [T, B, A]
            log_pi = (logp_all * pi).sum(-1, keepdim=True)  # [T, B, 1]
        q1 = self.q_net(BATCH.obs, BATCH.action, begin_mask=BATCH.begin_mask)  # [T, B, 1]
        q2 = self.q_net2(BATCH.obs, BATCH.action, begin_mask=BATCH.begin_mask)  # [T, B, 1]
        q1_pi = self.q_net(BATCH.obs, pi, begin_mask=BATCH.begin_mask)  # [T, B, 1]
        q2_pi = self.q_net2(BATCH.obs, pi, begin_mask=BATCH.begin_mask)  # [T, B, 1]
        dc_r = n_step_return(BATCH.reward,
                             self.gamma,
                             BATCH.done,
                             v_target,
                             BATCH.begin_mask).detach()  # [T, B, 1]
        v_from_q_stop = (th.minimum(q1_pi, q2_pi) - self.alpha * log_pi).detach()  # [T, B, 1]
        td_v = v - v_from_q_stop  # [T, B, 1]
        td_error1 = q1 - dc_r  # [T, B, 1]
        td_error2 = q2 - dc_r  # [T, B, 1]
        q1_loss = (td_error1.square() * BATCH.get('isw', 1.0)).mean()  # 1
        q2_loss = (td_error2.square() * BATCH.get('isw', 1.0)).mean()  # 1
        v_loss_stop = (td_v.square() * BATCH.get('isw', 1.0)).mean()  # 1

        critic_loss = 0.5 * q1_loss + 0.5 * q2_loss + 0.5 * v_loss_stop
        self.critic_oplr.optimize(critic_loss)

        if self.is_continuous:
            mu, log_std = self.actor(
                BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            pi = dist.rsample()  # [T, B, A]
            pi, log_pi = squash_action(
                pi, dist.log_prob(pi).unsqueeze(-1))  # [T, B, A], [T, B, 1]
            entropy = dist.entropy().mean()  # 1
        else:
            logits = self.actor(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
            logp_all = logits.log_softmax(-1)  # [T, B, A]
            gumbel_noise = td.Gumbel(0, 1).sample(logp_all.shape)  # [T, B, A]
            _pi = ((logp_all + gumbel_noise) / self.discrete_tau).softmax(-1)  # [T, B, A]
            _pi_true_one_hot = F.one_hot(
                _pi.argmax(-1), self.a_dim).float()  # [T, B, A]
            _pi_diff = (_pi_true_one_hot - _pi).detach()  # [T, B, A]
            pi = _pi_diff + _pi  # [T, B, A]
            log_pi = (logp_all * pi).sum(-1, keepdim=True)  # [T, B, 1]
            entropy = -(logp_all.exp() * logp_all).sum(-1).mean()  # 1
        q1_pi = self.q_net(BATCH.obs, pi, begin_mask=BATCH.begin_mask)  # [T, B, 1]
        actor_loss = -(q1_pi - self.alpha * log_pi).mean()  # 1
        self.actor_oplr.optimize(actor_loss)

        self._summary_collector.add('LEARNING_RATE', 'actor_lr', self.actor_oplr.lr)
        self._summary_collector.add('LEARNING_RATE', 'critic_lr', self.critic_oplr.lr)
        self._summary_collector.add('LOSS', 'actor_loss', actor_loss)
        self._summary_collector.add('LOSS', 'q1_loss', q1_loss)
        self._summary_collector.add('LOSS', 'q2_loss', q2_loss)
        self._summary_collector.add('LOSS', 'v_loss', v_loss_stop)
        self._summary_collector.add('LOSS', 'critic_loss', critic_loss)
        self._summary_collector.add('Statistics', 'log_alpha', self.log_alpha)
        self._summary_collector.add('Statistics', 'alpha', self.alpha)
        self._summary_collector.add('Statistics', 'entropy', entropy)
        self._summary_collector.add('Statistics', 'q', th.minimum(q1, q2))
        self._summary_collector.add('Statistics', 'v', v)

        if self.auto_adaption:
            alpha_loss = -(self.alpha * (log_pi.detach() + self.target_entropy)).mean()
            self.alpha_oplr.optimize(alpha_loss)
            self._summary_collector.add('LOSS', 'alpha_loss', alpha_loss)
            self._summary_collector.add('LEARNING_RATE', 'alpha_lr', self.alpha_oplr.lr)
        return (td_error1 + td_error2) / 2

    @iton
    def _train_discrete(self, BATCH):
        v = self.v_net(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, 1]
        v_target = self.v_net.t(BATCH.obs_, begin_mask=BATCH.begin_mask)  # [T, B, 1]

        q1_all = self.q_net(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
        q2_all = self.q_net2(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
        q1 = (q1_all * BATCH.action).sum(-1, keepdim=True)  # [T, B, 1]
        q2 = (q2_all * BATCH.action).sum(-1, keepdim=True)  # [T, B, 1]
        logits = self.actor(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
        logp_all = logits.log_softmax(-1)  # [T, B, A]

        dc_r = n_step_return(BATCH.reward,
                             self.gamma,
                             BATCH.done,
                             v_target,
                             BATCH.begin_mask).detach()  # [T, B, 1]
        td_v = v - (th.minimum(
            (logp_all.exp() * q1_all).sum(-1, keepdim=True),
            (logp_all.exp() * q2_all).sum(-1, keepdim=True)
        )).detach()  # [T, B, 1]
        td_error1 = q1 - dc_r  # [T, B, 1]
        td_error2 = q2 - dc_r  # [T, B, 1]

        q1_loss = (td_error1.square() * BATCH.get('isw', 1.0)).mean()  # 1
        q2_loss = (td_error2.square() * BATCH.get('isw', 1.0)).mean()  # 1
        v_loss_stop = (td_v.square() * BATCH.get('isw', 1.0)).mean()  # 1
        critic_loss = 0.5 * q1_loss + 0.5 * q2_loss + 0.5 * v_loss_stop
        self.critic_oplr.optimize(critic_loss)

        q1_all = self.q_net(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
        q2_all = self.q_net2(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
        logits = self.actor(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
        logp_all = logits.log_softmax(-1)  # [T, B, A]

        entropy = -(logp_all.exp() * logp_all).sum(-1, keepdim=True)  # [T, B, 1]
        q_all = th.minimum(q1_all, q2_all)  # [T, B, A]
        actor_loss = -((q_all - self.alpha * logp_all) * logp_all.exp()).sum(-1)  # [T, B, A] => [T, B]
        actor_loss = actor_loss.mean()  # 1
        self.actor_oplr.optimize(actor_loss)

        self._summary_collector.add('LEARNING_RATE', 'actor_lr', self.actor_oplr.lr)
        self._summary_collector.add('LEARNING_RATE', 'critic_lr', self.critic_oplr.lr)
        self._summary_collector.add('LOSS', 'actor_loss', actor_loss)
        self._summary_collector.add('LOSS', 'q1_loss', q1_loss)
        self._summary_collector.add('LOSS', 'q2_loss', q2_loss)
        self._summary_collector.add('LOSS', 'v_loss', v_loss_stop)
        self._summary_collector.add('LOSS', 'critic_loss', critic_loss)
        self._summary_collector.add('Statistics', 'log_alpha', self.log_alpha)
        self._summary_collector.add('Statistics', 'alpha', self.alpha)
        self._summary_collector.add('Statistics', 'entropy', entropy)
        self._summary_collector.add('Statistics', 'v', v)

        if self.auto_adaption:
            corr = (self.target_entropy - entropy).detach()  # [T, B, 1]
            # corr = ((logp_all - self.a_dim) * logp_all.exp()).sum(-1).detach()
            alpha_loss = -(self.alpha * corr)  # [T, B, 1]
            alpha_loss = alpha_loss.mean()  # 1
            self.alpha_oplr.optimize(alpha_loss)
            self._summary_collector.add('LOSS', 'alpha_loss', alpha_loss)
            self._summary_collector.add('LEARNING_RATE', 'alpha_lr', self.alpha_oplr.lr)
        return (td_error1 + td_error2) / 2

    def _after_train(self):
        super()._after_train()
        if self.annealing and not self.auto_adaption:
            self.log_alpha.copy_(self.alpha_annealing(self._cur_train_step).log())
        self.v_net.sync()
