#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as th
import torch.distributions as td

from rls.algorithms.base.sarl_on_policy import SarlOnPolicy
from rls.common.data import Data
from rls.common.decorator import iton
from rls.nn.models import ActorDct, ActorMuLogstd, CriticValue
from rls.nn.utils import OPLR
from rls.utils.np_utils import calculate_td_error, discounted_sum
from rls.utils.torch_utils import grads_flatten, set_from_flat_params


class NPG(SarlOnPolicy):
    """
    Natural Policy Gradient, NPG
    https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf
    """
    policy_mode = 'on-policy'

    def __init__(self,
                 agent_spec,

                 actor_step_size=0.5,
                 beta=1.0e-3,
                 lambda_=0.95,
                 cg_iters=10,
                 damping_coeff=0.1,
                 epsilon=0.2,
                 critic_lr=1e-3,
                 train_critic_iters=10,
                 network_settings={
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
        self.actor_step_size = actor_step_size
        self.beta = beta
        self.lambda_ = lambda_
        self._epsilon = epsilon
        self._cg_iters = cg_iters
        self._damping_coeff = damping_coeff
        self._train_critic_iters = train_critic_iters

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

        self.critic_oplr = OPLR(self.critic, critic_lr, **self._oplr_params)
        self._trainer_modules.update(actor=self.actor,
                                     critic=self.critic,
                                     critic_oplr=self.critic_oplr)

    @iton
    def select_action(self, obs):
        output = self.actor(obs, rnncs=self.rnncs)  # [B, A]
        self.rnncs_ = self.actor.get_rnncs()
        value = self.critic(obs, rnncs=self.rnncs)  # [B, 1]
        if self.is_continuous:
            mu, log_std = output  # [B, A]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            action = dist.sample().clamp(-1, 1)  # [B, A]
            log_prob = dist.log_prob(action).unsqueeze(-1)  # [B, 1]
        else:
            logits = output  # [B, A]
            logp_all = logits.log_softmax(-1)  # [B, A]
            norm_dist = td.Categorical(logits=logp_all)
            action = norm_dist.sample()  # [B,]
            log_prob = norm_dist.log_prob(action).unsqueeze(-1)  # [B, 1]
        acts_info = Data(action=action,
                         value=value,
                         log_prob=log_prob + th.finfo().eps)
        if self.use_rnn:
            acts_info.update(rnncs=self.rnncs)
        if self.is_continuous:
            acts_info.update(mu=mu, log_std=log_std)
        else:
            acts_info.update(logp_all=logp_all)
        return action, acts_info

    @iton
    def _get_value(self, obs, rnncs=None):
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

    @iton
    def _train(self, BATCH):
        output = self.actor(
            BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
        if self.is_continuous:
            mu, log_std = output  # [T, B, A], [T, B, A]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            new_log_prob = dist.log_prob(BATCH.action).unsqueeze(-1)  # [T, B, 1]
            entropy = dist.entropy().mean()  # 1
        else:
            logits = output  # [T, B, A]
            logp_all = logits.log_softmax(-1)  # [T, B, A]
            new_log_prob = (BATCH.action * logp_all).sum(-1, keepdim=True)  # [T, B, 1]
            entropy = -(logp_all.exp() * logp_all).sum(-1).mean()  # 1
        ratio = (new_log_prob - BATCH.log_prob).exp()  # [T, B, 1]
        actor_loss = -(ratio * BATCH.gae_adv).mean()  # 1

        flat_grads = grads_flatten(actor_loss, self.actor, retain_graph=True).detach()  # [1,]

        if self.is_continuous:
            kl = td.kl_divergence(
                td.Independent(td.Normal(BATCH.mu, BATCH.log_std.exp()), 1),
                td.Independent(td.Normal(mu, log_std.exp()), 1)
            ).mean()
        else:
            kl = (BATCH.logp_all.exp() * (BATCH.logp_all - logp_all)).sum(-1).mean()  # 1

        flat_kl_grad = grads_flatten(kl, self.actor, create_graph=True)
        search_direction = -self._conjugate_gradients(flat_grads, flat_kl_grad, cg_iters=self._cg_iters)  # [1,]

        with th.no_grad():
            flat_params = th.cat([param.data.view(-1) for param in self.actor.parameters()])
            new_flat_params = flat_params + self.actor_step_size * search_direction
            set_from_flat_params(self.actor, new_flat_params)

        for _ in range(self._train_critic_iters):
            value = self.critic(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, 1]
            td_error = BATCH.discounted_reward - value  # [T, B, 1]
            critic_loss = td_error.square().mean()  # 1
            self.critic_oplr.optimize(critic_loss)

        return {
            'LOSS/actor_loss': actor_loss,
            'LOSS/critic_loss': critic_loss,
            'Statistics/entropy': entropy.mean(),
            'LEARNING_RATE/critic_lr': self.critic_oplr.lr
        }

    def _conjugate_gradients(self,
                             flat_grads,
                             flat_kl_grad,
                             cg_iters: int = 10,
                             residual_tol: float = 1e-10):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """
        x = th.zeros_like(flat_grads)
        r, p = flat_grads.clone(), flat_grads.clone()
        # Note: should be 'r, p = b - MVP(x)', but for x=0, MVP(x)=0.
        # Change if doing warm start.
        rdotr = r.dot(r)
        for i in range(cg_iters):
            z = self._MVP(p, flat_kl_grad)
            alpha = rdotr / (p.dot(z) + th.finfo().eps)
            x += alpha * p
            r -= alpha * z
            new_rdotr = r.dot(r)
            if new_rdotr < residual_tol:
                break
            p = r + new_rdotr / rdotr * p
            rdotr = new_rdotr
        return x

    def _MVP(self, v, flat_kl_grad):
        """Matrix vector product."""
        # caculate second order gradient of kl with respect to theta
        kl_v = (flat_kl_grad * v).sum()
        mvp = grads_flatten(kl_v, self.actor, retain_graph=True).detach()
        mvp += max(0, self._damping_coeff) * v
        return mvp
