#!/usr/bin/env python3
# encoding: utf-8

import torch as th
import torch.distributions as td

from rls.algorithms.single.npg import NPG
from rls.common.decorator import iton
from rls.utils.torch_utils import grads_flatten, set_from_flat_params


class TRPO(NPG):
    """
    Trust Region Policy Optimization, https://arxiv.org/abs/1502.05477
    """
    policy_mode = 'on-policy'

    def __init__(self,
                 agent_spec,

                 delta=0.01,
                 backtrack_iters=10,
                 backtrack_coeff=0.8,
                 **kwargs):
        super().__init__(agent_spec=agent_spec, **kwargs)
        self._delta = delta
        self._backtrack_iters = backtrack_iters
        self._backtrack_coeff = backtrack_coeff

    @iton
    def _train(self, BATCH):
        output = self.actor(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
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

        mvp = self._MVP(search_direction, flat_kl_grad)
        # TODO: why < 0
        step_size = th.sqrt(2 * self._delta / (search_direction * mvp + th.finfo().eps).sum(0, keepdim=True))  # [1,]

        with th.no_grad():
            flat_params = th.cat([param.data.view(-1) for param in self.actor.parameters()])
            for i in range(self._backtrack_iters):
                new_flat_params = flat_params + step_size * search_direction * (self._backtrack_coeff ** i)
                set_from_flat_params(self.actor, new_flat_params)

                output = self.actor(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
                if self.is_continuous:
                    mu, log_std = output  # [T, B, A], [T, B, A]
                    dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
                    new_log_prob = dist.log_prob(BATCH.action).unsqueeze(-1)  # [T, B, 1]
                    kl = td.kl_divergence(
                        td.Independent(td.Normal(BATCH.mu, BATCH.log_std.exp()), 1),
                        td.Independent(td.Normal(mu, log_std.exp()), 1)
                    ).mean()
                else:
                    logits = output  # [T, B, A]
                    logp_all = logits.log_softmax(-1)  # [T, B, A]
                    new_log_prob = (BATCH.action * logp_all).sum(-1, keepdim=True)  # [T, B, 1]
                    kl = (BATCH.logp_all.exp() * (BATCH.logp_all - logp_all)).sum(-1).mean()  # 1
                # [T, B, 1]
                new_dratio = (new_log_prob - BATCH.log_prob).exp()
                new_actor_loss = -(new_dratio * BATCH.gae_adv).mean()  # 1

                if kl < self._delta and new_actor_loss < actor_loss:
                    break

        for _ in range(self._train_critic_iters):
            value = self.critic(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, 1]
            td_error = BATCH.discounted_reward - value  # [T, B, 1]
            critic_loss = td_error.square().mean()  # 1
            self.critic_oplr.optimize(critic_loss)

        self._summary_collector.add('LOSS', 'actor_loss', actor_loss)
        self._summary_collector.add('LOSS', 'critic_loss', critic_loss)
        self._summary_collector.add('Statistics', 'entropy', entropy)
        self._summary_collector.add('LEARNING_RATE', 'critic_lr', self.critic_oplr.lr)
