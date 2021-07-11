#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from torch import distributions as td
from typing import (Union,
                    List,
                    Dict,
                    NoReturn)
from dataclasses import dataclass

from rls.utils.torch_utils import (gaussian_clip_rsample,
                                   gaussian_likelihood_sum,
                                   gaussian_entropy)
from rls.algos.base.on_policy import On_Policy
from rls.common.specs import (ModelObservations,
                              Data,
                              BatchExperiences)
from rls.nn.models import (ActorCriticValueCts,
                           ActorCriticValueDct,
                           ActorMuLogstd,
                           ActorDct,
                           CriticValue)
from rls.nn.utils import OPLR
from rls.utils.converter import to_numpy
from rls.common.decorator import iTensor_oNumpy


@dataclass(eq=False)
class PPO_Store_BatchExperiences(BatchExperiences):
    value: np.ndarray
    log_prob: np.ndarray


@dataclass(eq=False)
class PPO_Train_BatchExperiences(Data):
    obs: ModelObservations
    action: np.ndarray
    value: np.ndarray
    log_prob: np.ndarray
    discounted_reward: np.ndarray
    gae_adv: np.ndarray


class PPO(On_Policy):
    '''
    Proximal Policy Optimization, https://arxiv.org/abs/1707.06347
    Emergence of Locomotion Behaviours in Rich Environments, http://arxiv.org/abs/1707.02286, DPPO
    '''

    def __init__(self,
                 envspec,

                 policy_epoch: int = 4,
                 value_epoch: int = 4,
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
                 max_grad_norm: float = 0.5,
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
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'critic': [32, 32]
                 },
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.ent_coef = ent_coef
        self.policy_epoch = policy_epoch
        self.value_epoch = value_epoch
        self.lambda_ = lambda_
        assert 0.0 <= lambda_ <= 1.0, "GAE lambda should be in [0, 1]."
        self.epsilon = epsilon
        self.use_vclip = use_vclip
        self.value_epsilon = value_epsilon
        self.share_net = share_net
        self.kl_reverse = kl_reverse
        self.kl_target = kl_target
        self.kl_alpha = kl_alpha
        self.kl_coef = t.tensor(kl_coef).float()
        self.extra_coef = extra_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.use_duel_clip = use_duel_clip
        self.duel_epsilon = duel_epsilon
        if self.use_duel_clip:
            assert -self.epsilon < self.duel_epsilon < self.epsilon, "duel_epsilon should be set in the range of (-epsilon, epsilon)."

        self.kl_cutoff = kl_target * kl_target_cutoff
        self.kl_stop = kl_target * kl_target_earlystop
        self.kl_low = kl_target * kl_beta[0]
        self.kl_high = kl_target * kl_beta[-1]

        self.use_kl_loss = use_kl_loss
        self.use_extra_loss = use_extra_loss
        self.use_early_stop = use_early_stop

        if self.share_net:
            if self.is_continuous:
                self.net = ActorCriticValueCts(self.rep_net.h_dim,
                                               output_shape=self.a_dim,
                                               network_settings=network_settings['share']['continuous'])
            else:
                self.net = ActorCriticValueDct(self.rep_net.h_dim,
                                               output_shape=self.a_dim,
                                               network_settings=network_settings['share']['discrete'])
            if self.max_grad_norm is not None:
                self.oplr = OPLR([self.net, self.rep_net], lr, clipnorm=self.max_grad_norm)
            else:
                self.oplr = OPLR([self.net, self.rep_net], lr)

            self._worker_modules.update(rep_net=self.rep_net,
                                        model=self.net)
            self._trainer_modules.update(self._worker_modules)
            self._trainer_modules.update(oplr=self.oplr)
        else:
            if self.is_continuous:
                self.actor = ActorMuLogstd(self.rep_net.h_dim,
                                           output_shape=self.a_dim,
                                           network_settings=network_settings['actor_continuous'])
            else:
                self.actor = ActorDct(self.rep_net.h_dim,
                                      output_shape=self.a_dim,
                                      network_settings=network_settings['actor_discrete'])
            self.critic = CriticValue(self.rep_net.h_dim,
                                      network_settings=network_settings['critic'])
            if self.max_grad_norm is not None:
                self.actor_oplr = OPLR(self.actor, actor_lr, clipnorm=self.max_grad_norm)
                self.critic_oplr = OPLR([self.critic, self.rep_net], critic_lr, clipnorm=self.max_grad_norm)
            else:
                self.actor_oplr = OPLR(self.actor, actor_lr)
                self.critic_oplr = OPLR([self.critic, self.rep_net], critic_lr)

            self._worker_modules.update(rep_net=self.rep_net,
                                        actor=self.actor)   # TODO
            self._trainer_modules.update(self._worker_modules)
            self._trainer_modules.update(critic=self.critic,
                                         actor_oplr=self.actor_oplr,
                                         critic_oplr=self.critic_oplr)

        self.initialize_data_buffer(store_data_type=PPO_Store_BatchExperiences,
                                    sample_data_type=PPO_Train_BatchExperiences)

    def __call__(self, obs, evaluation: bool = False) -> np.ndarray:
        a = self._get_action(obs)
        return a

    @iTensor_oNumpy
    def _get_action(self, obs):
        feat, self.next_cell_state = self.rep_net(obs, cell_state=self.cell_state)
        if self.is_continuous:
            if self.share_net:
                mu, log_std, value = self.net(feat)
            else:
                mu, log_std = self.actor(feat)
                value = self.critic(feat)
            sample_op, _ = gaussian_clip_rsample(mu, log_std)
            log_prob = gaussian_likelihood_sum(sample_op, mu, log_std)
        else:
            if self.share_net:
                logits, value = self.net(feat)
            else:
                logits = self.actor(feat)
                value = self.critic(feat)
            norm_dist = td.categorical.Categorical(logits=logits)
            sample_op = norm_dist.sample()
            log_prob = norm_dist.log_prob(sample_op)
        self._value = to_numpy(value)
        self._log_prob = to_numpy(log_prob) + 1e-10
        return sample_op

    def store_data(self, exps: BatchExperiences) -> NoReturn:
        # self._running_average()
        self.data.add(PPO_Store_BatchExperiences(*exps.astuple(), self._value, self._log_prob))
        if self.use_rnn:
            self.data.add_cell_state(tuple(cs.numpy() for cs in self.cell_state))
        self.cell_state = self.next_cell_state

    @iTensor_oNumpy
    def _get_value(self, obs, cell_state):
        feat, cell_state = self.rep_net(obs, cell_state=cell_state)
        if self.share_net:
            if self.is_continuous:
                _, _, value = self.net(feat)
            else:
                _, value = self.net(feat)
        else:
            value = self.critic(feat)
        return value, cell_state

    def calculate_statistics(self) -> NoReturn:
        init_value, self.cell_state = self._get_value(self.data.last_data().obs_, cell_state=self.cell_state)
        self.data.cal_dc_r(self.gamma, init_value)
        self.data.cal_td_error(self.gamma, init_value)
        self.data.cal_gae_adv(self.lambda_, self.gamma, normalize=True)

    def learn(self, **kwargs) -> NoReturn:
        self.train_step = kwargs.get('train_step')

        def _train(data, cell_state):
            early_step = 0
            if self.share_net:
                for i in range(self.policy_epoch):
                    actor_loss, critic_loss, entropy, kl = self.train_share(data, cell_state, self.kl_coef)
                    if self.use_early_stop and kl > self.kl_stop:
                        early_step = i
                        break
            else:
                for i in range(self.policy_epoch):
                    actor_loss, entropy, kl = self.train_actor(data, cell_state, self.kl_coef)
                    if self.use_early_stop and kl > self.kl_stop:
                        early_step = i
                        break

                for _ in range(self.value_epoch):
                    critic_loss = self.train_critic(data, cell_state)

            summaries = dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/kl', kl],
                ['Statistics/entropy', entropy]
            ])

            if self.use_early_stop:
                summaries.update(dict([
                    ['Statistics/early_step', early_step]
                ]))

            if self.use_kl_loss:
                # ref: https://github.com/joschu/modular_rl/blob/6970cde3da265cf2a98537250fea5e0c0d9a7639/modular_rl/ppo.py#L93
                if kl > self.kl_high:
                    self.kl_coef *= self.kl_alpha
                elif kl < self.kl_low:
                    self.kl_coef /= self.kl_alpha

                summaries.update(dict([
                    ['Statistics/kl_coef', self.kl_coef]
                ]))
            return summaries

        if self.share_net:
            summary_dict = dict([['LEARNING_RATE/lr', self.oplr.lr]])
        else:
            summary_dict = dict([
                ['LEARNING_RATE/actor_lr', self.actor_oplr.lr],
                ['LEARNING_RATE/critic_lr', self.critic_oplr.lr]
            ])

        self._learn(function_dict={
            'calculate_statistics': self.calculate_statistics,
            'train_function': _train,
            'summary_dict': summary_dict,
            'train_data_type': PPO_Train_BatchExperiences
        })

    @iTensor_oNumpy
    def train_share(self, BATCH, cell_state, kl_coef):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_state['obs'])
        if self.is_continuous:
            mu, log_std, value = self.net(feat)
            new_log_prob = gaussian_likelihood_sum(BATCH.action, mu, log_std)
            entropy = gaussian_entropy(log_std)
        else:
            logits, value = self.net(feat)
            logp_all = logits.log_softmax(-1)
            new_log_prob = (BATCH.action * logp_all).sum(1, keepdim=True)
            entropy = -(logp_all.exp() * logp_all).sum(1, keepdim=True).mean()
        ratio = (new_log_prob - BATCH.log_prob).exp()
        surrogate = ratio * BATCH.gae_adv
        clipped_surrogate = t.minimum(
            surrogate,
            ratio.clamp(1.0 - self.epsilon, 1.0 + self.epsilon) * BATCH.gae_adv
        )
        # ref: https://github.com/thu-ml/tianshou/blob/c97aa4065ee8464bd5897bb86f1f81abd8e2cff9/tianshou/policy/modelfree/ppo.py#L159
        if self.use_duel_clip:
            clipped_surrogate = t.maximum(
                clipped_surrogate,
                (1.0 + self.duel_epsilon) * BATCH.gae_adv
            )
        actor_loss = -(clipped_surrogate.mean() + self.ent_coef * entropy)

        # ref: https://github.com/joschu/modular_rl/blob/6970cde3da265cf2a98537250fea5e0c0d9a7639/modular_rl/ppo.py#L40
        # ref: https://github.com/hill-a/stable-baselines/blob/b3f414f4f2900403107357a2206f80868af16da3/stable_baselines/ppo2/ppo2.py#L185
        if self.kl_reverse:  # TODO:
            kl = .5 * (new_log_prob - BATCH.log_prob).square().mean()
        else:
            kl = .5 * (BATCH.log_prob - new_log_prob).square().mean()    # a sample estimate for KL-divergence, easy to compute

        td_error = BATCH.discounted_reward - value
        if self.use_vclip:
            # ref: https://github.com/llSourcell/OpenAI_Five_vs_Dota2_Explained/blob/c5def7e57aa70785c2394ea2eeb3e5f66ad59a53/train.py#L154
            # ref: https://github.com/hill-a/stable-baselines/blob/b3f414f4f2900403107357a2206f80868af16da3/stable_baselines/ppo2/ppo2.py#L172
            value_clip = BATCH.value + (value - BATCH.value).clamp(-self.value_epsilon, self.value_epsilon)
            td_error_clip = BATCH.discounted_reward - value_clip
            td_square = t.maximum(td_error.square(), td_error_clip.square())
        else:
            td_square = td_error.square()

        if self.use_kl_loss:
            kl_loss = kl_coef * kl
            actor_loss += kl_loss

        if self.use_extra_loss:
            extra_loss = self.extra_coef * t.maximum(t.zeros_like(kl), kl - self.kl_cutoff).square()
            actor_loss += extra_loss
        value_loss = 0.5 * td_square.mean()
        loss = actor_loss + self.vf_coef * value_loss
        self.oplr.step(loss)
        self.global_step.add_(1)
        return actor_loss, value_loss, entropy, kl

    @iTensor_oNumpy
    def train_actor(self, BATCH, cell_state, kl_coef):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_state['obs'])
        if self.is_continuous:
            mu, log_std = self.actor(feat)
            new_log_prob = gaussian_likelihood_sum(BATCH.action, mu, log_std)
            entropy = gaussian_entropy(log_std)
        else:
            logits = self.actor(feat)
            logp_all = logits.log_softmax(-1)
            new_log_prob = (BATCH.action * logp_all).sum(1, keepdim=True)
            entropy = -(logp_all.exp() * logp_all).sum(1, keepdim=True).mean()
        ratio = (new_log_prob - BATCH.log_prob).exp()
        kl = (BATCH.log_prob - new_log_prob).mean()
        surrogate = ratio * BATCH.gae_adv
        clipped_surrogate = t.minimum(
            surrogate,
            t.where(BATCH.gae_adv > 0, (1 + self.epsilon) * BATCH.gae_adv, (1 - self.epsilon) * BATCH.gae_adv)
        )
        if self.use_duel_clip:
            clipped_surrogate = t.maximum(
                clipped_surrogate,
                (1.0 + self.duel_epsilon) * BATCH.gae_adv
            )

        actor_loss = -(clipped_surrogate.mean() + self.ent_coef * entropy)

        if self.use_kl_loss:
            kl_loss = kl_coef * kl
            actor_loss += kl_loss
        if self.use_extra_loss:
            extra_loss = self.extra_coef * t.maximum(t.zeros_like(kl), kl - self.kl_cutoff).square()
            actor_loss += extra_loss

        self.actor_oplr.step(actor_loss)
        self.global_step.add_(1)
        return actor_loss, entropy, kl

    @iTensor_oNumpy
    def train_critic(self, BATCH, cell_state):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_state['obs'])
        value = self.critic(feat)

        td_error = BATCH.discounted_reward - value
        if self.use_vclip:
            value_clip = BATCH.value + (value - BATCH.value).clamp(-self.value_epsilon, self.value_epsilon)
            td_error_clip = BATCH.discounted_reward - value_clip
            td_square = t.maximum(td_error.square(), td_error_clip.square())
        else:
            td_square = td_error.square()

        value_loss = 0.5 * td_square.mean()
        self.critic_oplr.step(value_loss)
        return value_loss
