#!/usr/bin/env python3
# encoding: utf-8


import torch as t
from torch import distributions as td

from rls.algorithms.single.modelbased.dreamer_v1 import DreamerV1
from rls.nn.dreamer import RecurrentStateSpaceModel
from rls.nn.modules.wrappers import TargetTwin


class DreamerV2(DreamerV1):
    '''
    Mastering Atari with Discrete World Models, http://arxiv.org/abs/2010.02193
    '''
    policy_mode = 'off-policy'

    def __init__(self,

                 discretes=32,
                 kl_forward=False,
                 kl_balance=0.8,
                 kl_free_avg=True,
                 use_free_nats=False,
                 actor_grad='reinforce',
                 actor_entropy_scale=1e-3,
                 actor_grad_mix=0.,
                 use_double=True,
                 assign_interval=100,
                 network_settings=dict(),
                 **kwargs):
        self._discretes = discretes
        self.kl_forward = kl_forward
        self.kl_balance = kl_balance
        self.kl_free_avg = kl_free_avg
        self._use_free_nats = use_free_nats
        self.actor_grad = actor_grad
        self._actor_entropy_scale = actor_entropy_scale
        self._actor_grad_mix = actor_grad_mix
        self._use_double = use_double
        self.assign_interval = assign_interval
        super().__init__(network_settings=network_settings, **kwargs)

    @property
    def _action_dist(self):
        return 'trunc_normal' if self.is_continuous else 'one_hot'  # 'relaxed_one_hot'

    @property
    def decoder_input_dim(self):
        return self.deter_dim + self.stoch_dim * (1 if self._discretes <= 0 else self._discretes)

    def _dreamer_build_rssm(self):
        return RecurrentStateSpaceModel(self.stoch_dim,
                                        self.deter_dim,
                                        self.a_dim,
                                        self.obs_encoder.h_dim,
                                        discretes=self._discretes,
                                        **self._network_settings['rssm']).to(self.device)

    def _dreamer_build_critic(self):
        return TargetTwin(super()._dreamer_build_critic()).to(self.device)

    def _kl_loss(self, prior_dist, post_dist):
        if self.kl_balance == 0.5:
            if self._use_free_nats:
                loss = td.kl_divergence(prior_dist, post_dist).clamp(min=self.kl_free_nats).mean()  # 1
            else:
                loss = td.kl_divergence(prior_dist, post_dist).mean()  # 1
        else:
            prior_dist_detached = self.rssm.clone_dist(prior_dist, detach=True)
            post_dist_detached = self.rssm.clone_dist(post_dist, detach=True)
            value_lhs = td.kl_divergence(
                prior_dist, post_dist_detached)  # [B,]
            value_rhs = td.kl_divergence(
                prior_dist_detached, post_dist)  # [B,]
            if self._use_free_nats:
                if self.kl_free_avg:
                    loss_lhs = value_lhs.mean().clamp(min=self.kl_free_nats)  # 1
                    loss_rhs = value_rhs.mean().clamp(min=self.kl_free_nats)  # 1
                else:
                    loss_lhs = value_lhs.clamp(
                        min=self.kl_free_nats).mean()  # 1
                    loss_rhs = value_rhs.clamp(
                        min=self.kl_free_nats).mean()  # 1
            else:
                loss_lhs = value_lhs.mean()  # 1
                loss_rhs = value_rhs.mean()  # 1
            mix = self.kl_balance if self.kl_forward else (1 - self.kl_balance)
            loss = mix * loss_lhs + (1 - mix) * loss_rhs
        return loss

    def _dreamer_target_img_value(self, imaginated_feats):
        if self._use_double:
            imaginated_values = self.critic.t(
                imaginated_feats).mean  # [H, T*B, 1]
        else:
            imaginated_values = self.critic(
                imaginated_feats).mean  # [H, T*B, 1
        return imaginated_values

    def _dreamer_build_actor_loss(self, imaginated_feats, log_probs, entropies, discount, returns):
        if self.actor_grad == 'dynamics':
            objective = returns  # [H-1, T*B, 1]
        elif self.actor_grad == 'reinforce':
            baseline = self.critic(imaginated_feats[:-1]).mean  # [H-1, T*B, 1]
            advantage = (returns - baseline).detach()   # [H-1, T*B, 1]
            objective = log_probs[1:] * advantage    # [H-1, T*B, 1]   # detach
        elif self.actor_grad == 'both':
            baseline = self.critic(imaginated_feats[:-1]).mean  # [H-1, T*B, 1]
            advantage = (returns - baseline).detach()   # [H-1, T*B, 1]
            objective = log_probs[1:] * advantage    # [H-1, T*B, 1]
            objective = self._actor_grad_mix * returns + \
                (1. - self._actor_grad_mix) * objective
        else:
            raise NotImplementedError(self.actor_grad)
        objective += self._actor_entropy_scale * entropies[1:]  # [H-1, T*B, 1]
        actor_loss = -(discount * objective).mean()
        return actor_loss

    def _after_train(self):
        super()._after_train()
        if self.cur_train_step % self.assign_interval == 0:
            self.critic.sync()
