#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as th
import torch.nn.functional as F

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.common.data import Data
from rls.common.decorator import iton
from rls.nn.models import BCQ_DCT, BCQ_Act_Cts, BCQ_CriticQvalueOne
from rls.nn.modules.wrappers import TargetTwin
from rls.nn.offline.bcq_vae import VAE
from rls.nn.utils import OPLR
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import n_step_return


class BCQ(SarlOffPolicy):
    """
    Benchmarking Batch Deep Reinforcement Learning Algorithms, http://arxiv.org/abs/1910.01708
    Off-Policy Deep Reinforcement Learning without Exploration, http://arxiv.org/abs/1812.02900
    """
    policy_mode = 'off-policy'

    def __init__(self,
                 polyak=0.995,
                 discrete=dict(threshold=0.3,
                               lr=5.0e-4,
                               eps_init=1,
                               eps_mid=0.2,
                               eps_final=0.01,
                               init2mid_annealing_step=1000,
                               assign_interval=1000,
                               network_settings=[32, 32]),
                 continuous=dict(phi=0.05,
                                 lmbda=0.75,
                                 select_samples=100,
                                 train_samples=10,
                                 actor_lr=1e-3,
                                 critic_lr=1e-3,
                                 vae_lr=1e-3,
                                 network_settings=dict(actor=[32, 32],
                                                       critic=[32, 32],
                                                       vae=dict(encoder=[750, 750],
                                                                decoder=[750, 750]))),
                 **kwargs):
        super().__init__(**kwargs)
        self._polyak = polyak

        if self.is_continuous:
            self._lmbda = continuous['lmbda']
            self._select_samples = continuous['select_samples']
            self._train_samples = continuous['train_samples']
            self.actor = TargetTwin(BCQ_Act_Cts(self.obs_spec,
                                                rep_net_params=self._rep_net_params,
                                                action_dim=self.a_dim,
                                                phi=continuous['phi'],
                                                network_settings=continuous['network_settings']['actor']),
                                    polyak=self._polyak).to(self.device)
            self.critic = TargetTwin(BCQ_CriticQvalueOne(self.obs_spec,
                                                         rep_net_params=self._rep_net_params,
                                                         action_dim=self.a_dim,
                                                         network_settings=continuous['network_settings']['critic']),
                                     polyak=self._polyak).to(self.device)
            self.vae = VAE(self.obs_spec,
                           rep_net_params=self._rep_net_params,
                           a_dim=self.a_dim,
                           z_dim=self.a_dim * 2,
                           hiddens=continuous['network_settings']['vae']).to(self.device)

            self.actor_oplr = OPLR(self.actor, continuous['actor_lr'], **self._oplr_params)
            self.critic_oplr = OPLR(self.critic, continuous['critic_lr'], **self._oplr_params)
            self.vae_oplr = OPLR(self.vae, continuous['vae_lr'], **self._oplr_params)
            self._trainer_modules.update(actor=self.actor,
                                         critic=self.critic,
                                         vae=self.vae,
                                         actor_oplr=self.actor_oplr,
                                         critic_oplr=self.critic_oplr,
                                         vae_oplr=self.vae_oplr)
        else:
            self.expl_expt_mng = ExplorationExploitationClass(eps_init=discrete['eps_init'],
                                                              eps_mid=discrete['eps_mid'],
                                                              eps_final=discrete['eps_final'],
                                                              init2mid_annealing_step=discrete[
                                                                  'init2mid_annealing_step'],
                                                              max_step=self._max_train_step)
            self.assign_interval = discrete['assign_interval']
            self._threshold = discrete['threshold']
            self.q_net = TargetTwin(BCQ_DCT(self.obs_spec,
                                            rep_net_params=self._rep_net_params,
                                            output_shape=self.a_dim,
                                            network_settings=discrete['network_settings']),
                                    polyak=self._polyak).to(self.device)
            self.oplr = OPLR(self.q_net, discrete['lr'], **self._oplr_params)
            self._trainer_modules.update(model=self.q_net,
                                         oplr=self.oplr)

    @iton
    def select_action(self, obs):
        if self.is_continuous:
            _actions = []
            for _ in range(self._select_samples):
                _actions.append(self.actor(obs, self.vae.decode(obs), rnncs=self.rnncs))  # [B, A]
            self.rnncs_ = self.actor.get_rnncs()  # TODO: calculate corrected hidden state
            _actions = th.stack(_actions, dim=0)  # [N, B, A]
            q1s = []
            for i in range(self._select_samples):
                q1s.append(self.critic(obs, _actions[i])[0])
            q1s = th.stack(q1s, dim=0)  # [N, B, 1]
            max_idxs = q1s.argmax(dim=0, keepdim=True)[-1]  # [1, B, 1]
            actions = _actions[max_idxs, th.arange(self.n_copies).reshape(self.n_copies, 1), th.arange(self.a_dim)]
        else:
            q_values, i_values = self.q_net(obs, rnncs=self.rnncs)  # [B, *]
            q_values = q_values - q_values.min(dim=-1, keepdim=True)[0]  # [B, *]
            i_values = F.log_softmax(i_values, dim=-1)  # [B, *]
            i_values = i_values.exp()  # [B, *]
            i_values = (i_values / i_values.max(-1, keepdim=True)[0] > self._threshold).float()  # [B, *]

            self.rnncs_ = self.q_net.get_rnncs()

            if self._is_train_mode and self.expl_expt_mng.is_random(self._cur_train_step):
                actions = np.random.randint(0, self.a_dim, self.n_copies)
            else:
                actions = (i_values * q_values).argmax(-1)  # [B,]
        return actions, Data(action=actions)

    @iton
    def _train(self, BATCH):
        if self.is_continuous:
            # Variational Auto-Encoder Training
            recon, mean, std = self.vae(BATCH.obs, BATCH.action, begin_mask=BATCH.begin_mask)
            recon_loss = F.mse_loss(recon, BATCH.action)

            KL_loss = -0.5 * (1 + th.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            self.vae_oplr.optimize(vae_loss)

            target_Qs = []
            for _ in range(self._train_samples):
                # Compute value of perturbed actions sampled from the VAE
                _vae_actions = self.vae.decode(BATCH.obs_,
                                               begin_mask=BATCH.begin_mask)
                _actor_actions = self.actor.t(BATCH.obs_, _vae_actions,
                                              begin_mask=BATCH.begin_mask)
                target_Q1, target_Q2 = self.critic.t(BATCH.obs_, _actor_actions,
                                                     begin_mask=BATCH.begin_mask)

                # Soft Clipped Double Q-learning
                target_Q = self._lmbda * th.min(target_Q1, target_Q2) + \
                           (1. - self._lmbda) * th.max(target_Q1, target_Q2)
                target_Qs.append(target_Q)
            target_Qs = th.stack(target_Qs, dim=0)  # [N, T, B, 1]
            # Take max over each BATCH.action sampled from the VAE
            target_Q = target_Qs.max(dim=0)[0]  # [T, B, 1]

            target_Q = n_step_return(BATCH.reward,
                                     self.gamma,
                                     BATCH.done,
                                     target_Q,
                                     BATCH.begin_mask).detach()  # [T, B, 1]

            current_Q1, current_Q2 = self.critic(BATCH.obs, BATCH.action, begin_mask=BATCH.begin_mask)
            td_error = ((current_Q1 - target_Q) + (current_Q2 - target_Q)) / 2
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_oplr.optimize(critic_loss)

            # Pertubation Model / Action Training
            sampled_actions = self.vae.decode(BATCH.obs, begin_mask=BATCH.begin_mask)
            perturbed_actions = self.actor(BATCH.obs, sampled_actions, begin_mask=BATCH.begin_mask)

            # Update through DPG
            q1, _ = self.critic(BATCH.obs, perturbed_actions, begin_mask=BATCH.begin_mask)
            actor_loss = -q1.mean()

            self.actor_oplr.optimize(actor_loss)

            return td_error, {
                'LEARNING_RATE/actor_lr': self.actor_oplr.lr,
                'LEARNING_RATE/critic_lr': self.critic_oplr.lr,
                'LEARNING_RATE/vae_lr': self.vae_oplr.lr,
                'LOSS/actor_loss': actor_loss,
                'LOSS/critic_loss': critic_loss,
                'LOSS/vae_loss': vae_loss,
                'Statistics/q_min': q1.min(),
                'Statistics/q_mean': q1.mean(),
                'Statistics/q_max': q1.max()
            }

        else:
            q_next, i_next = self.q_net(BATCH.obs_, begin_mask=BATCH.begin_mask)  # [T, B, A]
            q_next = q_next - q_next.min(dim=-1, keepdim=True)[0]  # [B, *]
            i_next = F.log_softmax(i_next, dim=-1)  # [T, B, A]
            i_next = i_next.exp()  # [T, B, A]
            i_next = (i_next / i_next.max(-1, keepdim=True)[0] > self._threshold).float()  # [T, B, A]
            q_next = i_next * q_next  # [T, B, A]
            next_max_action = q_next.argmax(-1)  # [T, B]
            next_max_action_one_hot = F.one_hot(next_max_action.squeeze(), self.a_dim).float()  # [T, B, A]

            q_target_next, _ = self.q_net.t(BATCH.obs_, begin_mask=BATCH.begin_mask)  # [T, B, A]
            q_target_next_max = (q_target_next * next_max_action_one_hot).sum(-1, keepdim=True)  # [T, B, 1]
            q_target = n_step_return(BATCH.reward,
                                     self.gamma,
                                     BATCH.done,
                                     q_target_next_max,
                                     BATCH.begin_mask).detach()  # [T, B, 1]

            q, i = self.q_net(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
            q_eval = (q * BATCH.action).sum(-1, keepdim=True)  # [T, B, 1]

            td_error = q_target - q_eval  # [T, B, 1]
            q_loss = (td_error.square() * BATCH.get('isw', 1.0)).mean()  # 1

            imt = F.log_softmax(i, dim=-1)  # [T, B, A]
            imt = imt.reshape(-1, self.a_dim)  # [T*B, A]
            action = BATCH.action.reshape(-1, self.a_dim)  # [T*B, A]
            i_loss = F.nll_loss(imt, action.argmax(-1))  # 1

            loss = q_loss + i_loss + 1e-2 * i.pow(2).mean()

            self.oplr.optimize(loss)
            return td_error, {
                'LEARNING_RATE/lr': self.oplr.lr,
                'LOSS/q_loss': q_loss,
                'LOSS/i_loss': i_loss,
                'LOSS/loss': loss,
                'Statistics/q_max': q_eval.max(),
                'Statistics/q_min': q_eval.min(),
                'Statistics/q_mean': q_eval.mean()
            }

    def _after_train(self):
        super()._after_train()
        if self.is_continuous:
            self.actor.sync()
            self.critic.sync()
        else:
            if self._polyak != 0:
                self.q_net.sync()
            else:
                if self._cur_train_step % self.assign_interval == 0:
                    self.q_net.sync()
