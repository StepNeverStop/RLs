#!/usr/bin/env python3
# encoding: utf-8

from typing import Dict, List

import numpy as np
import torch as th
import torch.distributions as td

from rls.algorithms.base.sarl_on_policy import SarlOnPolicy
from rls.common.data import Data
from rls.common.decorator import iton
from rls.nn.models import (ActorCriticValueCts, ActorCriticValueDct, ActorDct,
                           ActorMuLogstd, CriticValue)
from rls.nn.utils import OPLR
from rls.utils.np_utils import calculate_td_error, discounted_sum


class PPO(SarlOnPolicy):
    """
    Proximal Policy Optimization, https://arxiv.org/abs/1707.06347
    Emergence of Locomotion Behaviours in Rich Environments, http://arxiv.org/abs/1707.02286, DPPO
    """
    policy_mode = 'on-policy'

    def __init__(self,
                 agent_spec,

                 ent_coef: float = 1.0e-2,
                 vf_coef: float = 0.5,
                 lr: float = 5.0e-4,
                 lambda_: float = 0.95,
                 epsilon: float = 0.2,
                 use_duel_clip: bool = False,
                 duel_epsilon: float = 0.,
                 use_vclip: bool = False,
                 value_epsilon: float = 0.2,
                 share_net: bool = True,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 1e-3,
                 kl_reverse: bool = False,
                 kl_target: float = 0.02,
                 kl_target_cutoff: float = 2,
                 kl_target_earlystop: float = 4,
                 kl_beta: List[float] = [0.7, 1.3],
                 kl_alpha: float = 1.5,
                 kl_coef: float = 1.0,
                 extra_coef: float = 1000.0,
                 use_kl_loss: bool = False,
                 use_extra_loss: bool = False,
                 use_early_stop: bool = False,
                 network_settings: Dict = {
                     'share': {
                         'continuous': {
                             'condition_sigma': False,
                             'log_std_bound': [-20, 2],
                             'share': [32, 32],
                             'mu': [32, 32],
                             'v': [32, 32]
                         },
                         'discrete': {
                             'share': [32, 32],
                             'logits': [32, 32],
                             'v': [32, 32]
                         }
                     },
                     'actor_continuous': {
                         'hidden_units': [64, 64],
                         'condition_sigma': False,
                         'log_std_bound': [-20, 2]
                     },
                     'actor_discrete': [32, 32],
                     'critic': [32, 32]
                 },
                 **kwargs):
        super().__init__(agent_spec=agent_spec, **kwargs)
        self._ent_coef = ent_coef
        self.lambda_ = lambda_
        assert 0.0 <= lambda_ <= 1.0, "GAE lambda should be in [0, 1]."
        self._epsilon = epsilon
        self._use_vclip = use_vclip
        self._value_epsilon = value_epsilon
        self._share_net = share_net
        self._kl_reverse = kl_reverse
        self._kl_target = kl_target
        self._kl_alpha = kl_alpha
        self._kl_coef = kl_coef
        self._extra_coef = extra_coef
        self._vf_coef = vf_coef

        self._use_duel_clip = use_duel_clip
        self._duel_epsilon = duel_epsilon
        if self._use_duel_clip:
            assert - \
                       self._epsilon < self._duel_epsilon < self._epsilon, "duel_epsilon should be set in the range of (-epsilon, epsilon)."

        self._kl_cutoff = kl_target * kl_target_cutoff
        self._kl_stop = kl_target * kl_target_earlystop
        self._kl_low = kl_target * kl_beta[0]
        self._kl_high = kl_target * kl_beta[-1]

        self._use_kl_loss = use_kl_loss
        self._use_extra_loss = use_extra_loss
        self._use_early_stop = use_early_stop

        if self._share_net:
            if self.is_continuous:
                self.net = ActorCriticValueCts(self.obs_spec,
                                               rep_net_params=self._rep_net_params,
                                               output_shape=self.a_dim,
                                               network_settings=network_settings['share']['continuous']).to(self.device)
            else:
                self.net = ActorCriticValueDct(self.obs_spec,
                                               rep_net_params=self._rep_net_params,
                                               output_shape=self.a_dim,
                                               network_settings=network_settings['share']['discrete']).to(self.device)
            self.oplr = OPLR(self.net, lr, **self._oplr_params)
            self._trainer_modules.update(model=self.net,
                                         oplr=self.oplr)
        else:
            if self.is_continuous:
                self.actor = ActorMuLogstd(self.obs_spec,
                                           rep_net_params=self._rep_net_params,
                                           output_shape=self.a_dim,
                                           network_settings=network_settings['actor_continuous']).to(self.device)
            else:
                self.actor = ActorDct(self.obs_spec,
                                      rep_net_params=self._rep_net_params,
                                      output_shape=self.a_dim,
                                      network_settings=network_settings['actor_discrete']).to(self.device)
            self.critic = CriticValue(self.obs_spec,
                                      rep_net_params=self._rep_net_params,
                                      network_settings=network_settings['critic']).to(self.device)
            self.actor_oplr = OPLR(self.actor, actor_lr, **self._oplr_params)
            self.critic_oplr = OPLR(self.critic, critic_lr, **self._oplr_params)
            self._trainer_modules.update(actor=self.actor,
                                         critic=self.critic,
                                         actor_oplr=self.actor_oplr,
                                         critic_oplr=self.critic_oplr)

    @iton
    def select_action(self, obs):
        if self.is_continuous:
            if self._share_net:
                mu, log_std, value = self.net(obs, rnncs=self.rnncs)  # [B, A]
                self.rnncs_ = self.net.get_rnncs()
            else:
                mu, log_std = self.actor(obs, rnncs=self.rnncs)  # [B, A]
                self.rnncs_ = self.actor.get_rnncs()
                value = self.critic(obs, rnncs=self.rnncs)  # [B, 1]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            action = dist.sample().clamp(-1, 1)  # [B, A]
            log_prob = dist.log_prob(action).unsqueeze(-1)  # [B, 1]
        else:
            if self._share_net:
                logits, value = self.net(obs, rnncs=self.rnncs)  # [B, A], [B, 1]
                self.rnncs_ = self.net.get_rnncs()
            else:
                logits = self.actor(obs, rnncs=self.rnncs)  # [B, A]
                self.rnncs_ = self.actor.get_rnncs()
                value = self.critic(obs, rnncs=self.rnncs)  # [B, 1]
            norm_dist = td.Categorical(logits=logits)
            action = norm_dist.sample()  # [B,]
            log_prob = norm_dist.log_prob(action).unsqueeze(-1)  # [B, 1]

        acts_info = Data(action=action,
                         value=value,
                         log_prob=log_prob + th.finfo().eps)
        if self.use_rnn:
            acts_info.update(rnncs=self.rnncs)
        return action, acts_info

    @iton
    def _get_value(self, obs, rnncs=None):
        if self._share_net:
            if self.is_continuous:
                _, _, value = self.net(obs, rnncs=rnncs)  # [B, 1]
            else:
                _, value = self.net(obs, rnncs=rnncs)  # [B, 1]
        else:
            value = self.critic(obs, rnncs=rnncs)  # [B, 1]
        return value

    def _preprocess_BATCH(self, BATCH):  # [T, B, *]
        BATCH = super()._preprocess_BATCH(BATCH)
        value = self._get_value(BATCH.obs_[-1], rnncs=self.rnncs)
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
                                       self.lambda_ * self.gamma,
                                       BATCH.done,
                                       BATCH.begin_mask,
                                       init_value=0.,
                                       normalize=True)
        return BATCH

    def learn(self, BATCH: Data):
        BATCH = self._preprocess_BATCH(BATCH)  # [T, B, *]
        for _ in range(self._epochs):
            kls = []
            for _BATCH in BATCH.sample(self._chunk_length, self.batch_size, repeat=self._sample_allow_repeat):
                _BATCH = self._before_train(_BATCH)
                summaries, kl = self._train(_BATCH)
                kls.append(kl)
                self.summaries.update(summaries)
                self._after_train()
            if self._use_early_stop and sum(kls) / len(kls) > self._kl_stop:
                break

    def _train(self, BATCH):
        if self._share_net:
            summaries, kl = self.train_share(BATCH)
        else:
            summaries = dict()
            actor_summaries, kl = self.train_actor(BATCH)
            critic_summaries = self.train_critic(BATCH)
            summaries.update(actor_summaries)
            summaries.update(critic_summaries)

        if self._use_kl_loss:
            # ref: https://github.com/joschu/modular_rl/blob/6970cde3da265cf2a98537250fea5e0c0d9a7639/modular_rl/ppo.py#L93
            if kl > self._kl_high:
                self._kl_coef *= self._kl_alpha
            elif kl < self._kl_low:
                self._kl_coef /= self._kl_alpha
            summaries.update({
                'Statistics/kl_coef': self._kl_coef
            })

        return summaries, kl

    @iton
    def train_share(self, BATCH):
        if self.is_continuous:
            # [T, B, A], [T, B, A], [T, B, 1]
            mu, log_std, value = self.net(BATCH.obs, begin_mask=BATCH.begin_mask)
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            new_log_prob = dist.log_prob(BATCH.action).unsqueeze(-1)  # [T, B, 1]
            entropy = dist.entropy().unsqueeze(-1)  # [T, B, 1]
        else:
            # [T, B, A], [T, B, 1]
            logits, value = self.net(BATCH.obs, begin_mask=BATCH.begin_mask)
            logp_all = logits.log_softmax(-1)  # [T, B, 1]
            new_log_prob = (BATCH.action * logp_all).sum(-1, keepdim=True)  # [T, B, 1]
            entropy = -(logp_all.exp() * logp_all).sum(-1, keepdim=True)  # [T, B, 1]
        ratio = (new_log_prob - BATCH.log_prob).exp()  # [T, B, 1]
        surrogate = ratio * BATCH.gae_adv  # [T, B, 1]
        clipped_surrogate = th.minimum(
            surrogate,
            ratio.clamp(1.0 - self._epsilon, 1.0 + self._epsilon) * BATCH.gae_adv
        )  # [T, B, 1]
        # ref: https://github.com/thu-ml/tianshou/blob/c97aa4065ee8464bd5897bb86f1f81abd8e2cff9/tianshou/policy/modelfree/ppo.py#L159
        if self._use_duel_clip:
            clipped_surrogate2 = th.maximum(
                clipped_surrogate,
                (1.0 + self._duel_epsilon) * BATCH.gae_adv
            )  # [T, B, 1]
            clipped_surrogate = th.where(BATCH.gae_adv < 0, clipped_surrogate2, clipped_surrogate)  # [T, B, 1]
        actor_loss = -(clipped_surrogate + self._ent_coef * entropy).mean()  # 1

        # ref: https://github.com/joschu/modular_rl/blob/6970cde3da265cf2a98537250fea5e0c0d9a7639/modular_rl/ppo.py#L40
        # ref: https://github.com/hill-a/stable-baselines/blob/b3f414f4f2900403107357a2206f80868af16da3/stable_baselines/ppo2/ppo2.py#L185
        if self._kl_reverse:  # TODO:
            kl = .5 * (new_log_prob - BATCH.log_prob).square().mean()  # 1
        else:
            # a sample estimate for KL-divergence, easy to compute
            kl = .5 * (BATCH.log_prob - new_log_prob).square().mean()

        if self._use_kl_loss:
            kl_loss = self._kl_coef * kl  # 1
            actor_loss += kl_loss

        if self._use_extra_loss:
            extra_loss = self._extra_coef * th.maximum(th.zeros_like(kl), kl - self._kl_cutoff).square().mean()  # 1
            actor_loss += extra_loss

        td_error = BATCH.discounted_reward - value  # [T, B, 1]
        if self._use_vclip:
            # ref: https://github.com/llSourcell/OpenAI_Five_vs_Dota2_Explained/blob/c5def7e57aa70785c2394ea2eeb3e5f66ad59a53/train.py#L154
            # ref: https://github.com/hill-a/stable-baselines/blob/b3f414f4f2900403107357a2206f80868af16da3/stable_baselines/ppo2/ppo2.py#L172
            value_clip = BATCH.value + (value - BATCH.value).clamp(-self._value_epsilon,
                                                                   self._value_epsilon)  # [T, B, 1]
            td_error_clip = BATCH.discounted_reward - value_clip  # [T, B, 1]
            td_square = th.maximum(td_error.square(), td_error_clip.square())  # [T, B, 1]
        else:
            td_square = td_error.square()  # [T, B, 1]

        critic_loss = 0.5 * td_square.mean()  # 1
        loss = actor_loss + self._vf_coef * critic_loss  # 1
        self.oplr.optimize(loss)
        return {
                   'LOSS/actor_loss': actor_loss,
                   'LOSS/critic_loss': critic_loss,
                   'Statistics/kl': kl,
                   'Statistics/entropy': entropy.mean(),
                   'LEARNING_RATE/lr': self.oplr.lr
               }, kl

    @iton
    def train_actor(self, BATCH):
        if self.is_continuous:
            # [T, B, A], [T, B, A]
            mu, log_std = self.actor(BATCH.obs, begin_mask=BATCH.begin_mask)
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            new_log_prob = dist.log_prob(BATCH.action).unsqueeze(-1)  # [T, B, 1]
            entropy = dist.entropy().unsqueeze(-1)  # [T, B, 1]
        else:
            logits = self.actor(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
            logp_all = logits.log_softmax(-1)  # [T, B, A]
            new_log_prob = (BATCH.action * logp_all).sum(-1, keepdim=True)  # [T, B, 1]
            entropy = -(logp_all.exp() * logp_all).sum(-1, keepdim=True)  # [T, B, 1]
        ratio = (new_log_prob - BATCH.log_prob).exp()  # [T, B, 1]
        kl = (BATCH.log_prob - new_log_prob).square().mean()  # 1
        surrogate = ratio * BATCH.gae_adv  # [T, B, 1]
        clipped_surrogate = th.minimum(
            surrogate,
            th.where(BATCH.gae_adv > 0, (1 + self._epsilon) *
                     BATCH.gae_adv, (1 - self._epsilon) * BATCH.gae_adv)
        )  # [T, B, 1]
        if self._use_duel_clip:
            clipped_surrogate = th.maximum(
                clipped_surrogate,
                (1.0 + self._duel_epsilon) * BATCH.gae_adv
            )  # [T, B, 1]

        actor_loss = -(clipped_surrogate + self._ent_coef * entropy).mean()  # 1

        if self._use_kl_loss:
            kl_loss = self._kl_coef * kl  # 1
            actor_loss += kl_loss
        if self._use_extra_loss:
            extra_loss = self._extra_coef * th.maximum(th.zeros_like(kl), kl - self._kl_cutoff).square().mean()  # 1
            actor_loss += extra_loss

        self.actor_oplr.optimize(actor_loss)
        return {
                   'LOSS/actor_loss': actor_loss,
                   'Statistics/kl': kl,
                   'Statistics/entropy': entropy.mean(),
                   'LEARNING_RATE/actor_lr': self.actor_oplr.lr
               }, kl

    @iton
    def train_critic(self, BATCH):
        value = self.critic(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, 1]

        td_error = BATCH.discounted_reward - value  # [T, B, 1]
        if self._use_vclip:
            value_clip = BATCH.value + (value - BATCH.value).clamp(-self._value_epsilon,
                                                                   self._value_epsilon)  # [T, B, 1]
            td_error_clip = BATCH.discounted_reward - value_clip  # [T, B, 1]
            td_square = th.maximum(td_error.square(), td_error_clip.square())  # [T, B, 1]
        else:
            td_square = td_error.square()  # [T, B, 1]

        critic_loss = 0.5 * td_square.mean()  # 1
        self.critic_oplr.optimize(critic_loss)
        return {
            'LOSS/critic_loss': critic_loss,
            'LEARNING_RATE/critic_lr': self.critic_oplr.lr
        }
