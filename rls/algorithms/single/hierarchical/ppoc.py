#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from torch import distributions as td

from rls.algorithms.base.sarl_on_policy import SarlOnPolicy
from rls.common.specs import Data
from rls.nn.models import PpocShare
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy
from rls.utils.np_utils import (discounted_sum,
                                calculate_td_error,
                                int2one_hot)


class PPOC(SarlOnPolicy):
    '''
    Learnings Options End-to-End for Continuous Action Tasks, PPOC, http://arxiv.org/abs/1712.00004
    '''
    policy_mode = 'on-policy'

    def __init__(self,
                 agent_spec,

                 options_num=4,
                 dc=0.01,
                 terminal_mask=False,
                 o_beta=1.0e-3,
                 pi_beta=1.0e-3,
                 lr=5.0e-4,
                 lambda_=0.95,
                 epsilon=0.2,
                 value_epsilon=0.2,
                 kl_reverse=False,
                 kl_target=0.02,
                 kl_target_cutoff=2,
                 kl_target_earlystop=4,
                 kl_beta=[0.7, 1.3],
                 kl_alpha=1.5,
                 kl_coef=1.0,
                 network_settings={
                     'share': [32, 32],
                     'q': [32, 32],
                     'intra_option': [32, 32],
                     'termination': [32, 32],
                     'o': [32, 32]
                 },
                 **kwargs):
        super().__init__(agent_spec=agent_spec, **kwargs)
        self.pi_beta = pi_beta
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.value_epsilon = value_epsilon
        self.kl_reverse = kl_reverse
        self.kl_target = kl_target
        self.kl_alpha = kl_alpha
        self.kl_coef = kl_coef

        self.kl_cutoff = kl_target * kl_target_cutoff
        self.kl_stop = kl_target * kl_target_earlystop
        self.kl_low = kl_target * kl_beta[0]
        self.kl_high = kl_target * kl_beta[-1]

        self.options_num = options_num
        self.dc = dc
        self.terminal_mask = terminal_mask
        self.o_beta = o_beta

        self.net = PpocShare(self.obs_spec,
                             rep_net_params=self._rep_net_params,
                             action_dim=self.a_dim,
                             options_num=self.options_num,
                             network_settings=network_settings,
                             is_continuous=self.is_continuous).to(self.device)

        if self.is_continuous:
            self.log_std = t.as_tensor(np.full(
                (self.options_num, self.a_dim), -0.5)).requires_grad_().to(self.device)  # [P, A]
            self.oplr = OPLR([self.net, self.log_std], lr)
        else:
            self.oplr = OPLR(self.net, lr)
        self._trainer_modules.update(model=self.net,
                                     oplr=self.oplr)

        self.oc_mask = t.tensor(np.zeros(self.n_copys)).to(self.device)
        self.options = t.tensor(np.random.randint(
            0, self.options_num, self.n_copys)).to(self.device)

    def episode_reset(self):
        super().episode_reset()
        self._done_mask = t.tensor(np.full(self.n_copys, True)).to(self.device)

    def episode_step(self, done: np.ndarray):  # TODO:
        super().episode_step(done)
        self._done_mask = t.tensor(done).to(self.device)
        self.options = self.new_options
        self.oc_mask = t.zeros_like(self.oc_mask)

    @iTensor_oNumpy
    def select_action(self, obs):
        # [B, P], [B, P, A], [B, P], [B, P]
        (q, pi, beta, o) = self.net(obs, cell_state=self.cell_state)
        self.next_cell_state = self.net.get_cell_state()
        options_onehot = t.nn.functional.one_hot(
            self.options, self.options_num).float()    # [B, P]
        options_onehot_expanded = options_onehot.unsqueeze(-1)  # [B, P, 1]
        pi = (pi * options_onehot_expanded).sum(-2)  # [B, A]
        if self.is_continuous:
            mu = pi  # [B, A]
            log_std = self.log_std[self.options]    # [B, A]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            action = dist.sample().clamp(-1, 1)  # [B, A]
            log_prob = dist.log_prob(action).unsqueeze(-1)   # [B, 1]
        else:
            logits = pi  # [B, A]
            norm_dist = td.Categorical(logits=logits)
            action = norm_dist.sample()  # [B,]
            log_prob = norm_dist.log_prob(action).unsqueeze(-1)    # [B, 1]
        o_log_prob = (o * options_onehot).sum(-1, keepdim=True)   # [B, 1]
        q_o = (q * options_onehot).sum(-1, keepdim=True)  # [B, 1]
        beta_adv = q_o - (q * o.exp()).sum(-1, keepdim=True)   # [B, 1]
        option_norm_dist = td.Categorical(logits=o)
        sample_options = option_norm_dist.sample()  # [B,]
        max_options = q.argmax(-1)  # [B, P] => [B, ]
        beta_probs = (beta * options_onehot).sum(-1)   # [B, P] => [B,]
        beta_dist = td.Bernoulli(probs=beta_probs)
        # <1 则不改变op， =1 则改变op
        new_options = t.where(beta_dist.sample() < 1,
                              self.options, sample_options)
        self.new_options = t.where(self._done_mask, max_options, new_options)
        self.oc_mask = (self.new_options == self.options).float()

        acts = Data(action=action,
                    value=q_o,
                    log_prob=log_prob+t.finfo().eps,
                    o_log_prob=o_log_prob+t.finfo().eps,
                    beta_advantage=beta_adv+self.dc,
                    last_options=self.options,
                    options=self.new_options,
                    reward_offset=-((1 - self.oc_mask) * self.dc).unsqueeze(-1))
        if self.use_rnn:
            acts.update(cell_state=self.cell_state)
        return action, acts

    @iTensor_oNumpy
    def _get_value(self, obs, options):
        (q, _, _, _) = self.net(obs, cell_state=self.cell_state)    # [T, B, P]
        value = (q * options).sum(-1, keepdim=True)  # [T, B, 1]
        return value

    def _preprocess_BATCH(self, BATCH):  # [T, B, *]
        BATCH = super()._preprocess_BATCH(BATCH)
        BATCH.reward += BATCH.reward_offset

        BATCH.last_options = int2one_hot(BATCH.last_options, self.options_num)
        BATCH.options = int2one_hot(BATCH.options, self.options_num)
        value = self._get_value(BATCH.obs_[-1], BATCH.options[-1])
        BATCH.discounted_reward = discounted_sum(BATCH.reward,
                                                 self.gamma,
                                                 BATCH.done,
                                                 BATCH.begin_mask,
                                                 init_value=value)
        td_error = calculate_td_error(BATCH.reward,
                                      self.gamma,
                                      BATCH.done,
                                      value=BATCH.value,
                                      next_value=np.concatenate((BATCH.value[1:], value[np.newaxis, :]), 0))
        BATCH.gae_adv = discounted_sum(td_error,
                                       self.lambda_*self.gamma,
                                       BATCH.done,
                                       BATCH.begin_mask,
                                       init_value=0.,
                                       normalize=True)
        return BATCH

    def learn(self, BATCH: Data):
        BATCH = self._preprocess_BATCH(BATCH)   # [T, B, *]
        for _ in range(self.epochs):
            kls = []
            for _BATCH in self._generate_BATCH(BATCH):
                _BATCH = self._before_train(_BATCH)
                summaries, kl = self._train(_BATCH)
                kls.append(kl)
                self.summaries.update(summaries)
                self._after_train()
            if sum(kls)/len(kls) > self.kl_stop:
                break

    @iTensor_oNumpy
    def _train(self, BATCH):
        # [T, B, P], [T, B, P, A], [T, B, P], [T, B, P]
        (q, pi, beta, o) = self.net(BATCH.obs)
        options_onehot_expanded = BATCH.options.unsqueeze(-1)  # [T, B, P, 1]

        # [T, B, P, A] => [T, B, A]
        pi = (pi * options_onehot_expanded).sum(-2)
        value = (q * BATCH.options).sum(-2, keepdim=True)    # [T, B, 1]

        if self.is_continuous:
            mu = pi  # [T, B, A]
            log_std = self.log_std[BATCH.options.argmax(-1)]    # [T, B, A]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            new_log_prob = dist.log_prob(
                BATCH.action).unsqueeze(-1)    # [T, B, 1]
            entropy = dist.entropy().mean()  # 1
        else:
            logits = pi  # [T, B, A]
            logp_all = logits.log_softmax(-1)   # [T, B, A]
            new_log_prob = (BATCH.action * logp_all).sum(-1,
                                                         keepdim=True)   # [T, B, 1]
            entropy = -(logp_all.exp() * logp_all).sum(-1,
                                                       keepdim=True).mean()  # 1
        ratio = (new_log_prob - BATCH.log_prob).exp()   # [T, B, 1]

        if self.kl_reverse:
            kl = (new_log_prob - BATCH.log_prob).mean()  # 1
        else:
            # a sample estimate for KL-divergence, easy to compute
            kl = (BATCH.log_prob - new_log_prob).mean()
        surrogate = ratio * BATCH.gae_adv   # [T, B, 1]

        value_clip = BATCH.value + \
            (value - BATCH.value).clamp(-self.value_epsilon,
                                        self.value_epsilon)  # [T, B, 1]
        td_error = BATCH.discounted_reward - value  # [T, B, 1]
        td_error_clip = BATCH.discounted_reward - value_clip    # [T, B, 1]
        td_square = t.maximum(
            td_error.square(), td_error_clip.square())    # [T, B, 1]

        pi_loss = -t.minimum(
            surrogate,
            ratio.clamp(1.0 - self.epsilon, 1.0 + self.epsilon) * BATCH.gae_adv
        ).mean()    # 1
        kl_loss = self.kl_coef * kl
        extra_loss = 1000.0 * \
            t.maximum(t.zeros_like(kl), kl - self.kl_cutoff).square().mean()
        pi_loss = pi_loss + kl_loss + extra_loss
        q_loss = 0.5 * td_square.mean()

        beta_s = (beta * BATCH.last_options).sum(-1, keepdim=True)   # [B, 1]
        beta_loss = beta_s * BATCH.beta_advantage   # [T, B, 1]
        if self.terminal_mask:
            beta_loss *= (1 - BATCH.done)  # [T, B, 1]
        beta_loss = beta_loss.mean()

        o_log_prob = (o * BATCH.options).sum(-1, keepdim=True)   # [T, B, 1]
        o_ratio = (o_log_prob - BATCH.o_log_prob).exp()  # [T, B, 1]
        o_entropy = -((o.exp() * o).sum(-1, keepdim=True)).mean()    # 1
        o_loss = -t.minimum(
            o_ratio * BATCH.gae_adv,
            o_ratio.clamp(1.0 - self.epsilon, 1.0 +
                          self.epsilon) * BATCH.gae_adv
        ).mean()    # 1

        loss = pi_loss + 1.0 * q_loss + o_loss + beta_loss - \
            self.pi_beta * entropy - self.o_beta * o_entropy   # 1
        self.oplr.step(loss)

        if kl > self.kl_high:
            self.kl_coef *= self.kl_alpha
        elif kl < self.kl_low:
            self.kl_coef /= self.kl_alpha

        return dict([
            ['LOSS/loss', loss],
            ['LOSS/pi_loss', pi_loss],
            ['LOSS/q_loss', q_loss],
            ['LOSS/o_loss', o_loss],
            ['LOSS/beta_loss', beta_loss],
            ['Statistics/kl', kl],
            ['Statistics/entropy', entropy],
            ['Statistics/o_entropy', o_entropy],
            ['Statistics/kl_coef', self.kl_coef],
            ['LEARNING_RATE/lr', self.oplr.lr]
        ]), kl
