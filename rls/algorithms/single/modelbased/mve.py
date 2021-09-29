#!/usr/bin/env python3
# encoding: utf-8

from copy import deepcopy

import torch as th
import torch.distributions as td
import torch.nn.functional as F

from rls.algorithms.single.ddpg import DDPG
from rls.common.data import get_first_vector
from rls.common.decorator import iton
from rls.nn.modelbased.done_model import VectorSA2D
from rls.nn.modelbased.forward_model import VectorSA2S
from rls.nn.modelbased.reward_model import VectorSA2R
from rls.nn.utils import OPLR


class MVE(DDPG):
    """
    Model-Based Value Estimation for Efficient Model-Free Reinforcement Learning, http://arxiv.org/abs/1803.00101
    """
    policy_mode = 'off-policy'

    def __init__(self,
                 wm_lr=1e-3,
                 roll_out_horizon=15,
                 **kwargs):
        super().__init__(**kwargs)
        network_settings = kwargs.get('network_settings', {})
        assert not self.obs_spec.has_visual_observation, "assert not self.obs_spec.has_visual_observation"
        assert self.obs_spec.has_vector_observation, "assert self.obs_spec.has_vector_observation"

        self._wm_lr = wm_lr
        self._roll_out_horizon = roll_out_horizon
        self._forward_dynamic_model = VectorSA2S(self.obs_spec.vector_dims[0],
                                                 self.a_dim,
                                                 hidden_units=network_settings['forward_model'])
        self._reward_model = VectorSA2R(self.obs_spec.vector_dims[0],
                                        self.a_dim,
                                        hidden_units=network_settings['reward_model'])
        self._done_model = VectorSA2D(self.obs_spec.vector_dims[0],
                                      self.a_dim,
                                      hidden_units=network_settings['done_model'])
        self._wm_oplr = OPLR([self._forward_dynamic_model, self._reward_model, self._done_model],
                             self._wm_lr, **self._oplr_params)
        self._trainer_modules.update(_forward_dynamic_model=self._forward_dynamic_model,
                                     _reward_model=self._reward_model,
                                     _done_model=self._done_model,
                                     _wm_oplr=self._wm_oplr)

    @iton
    def _train(self, BATCH):

        obs = get_first_vector(BATCH.obs)  # [T, B, S]
        obs_ = get_first_vector(BATCH.obs_)  # [T, B, S]
        _timestep = obs.shape[0]
        _batchsize = obs.shape[1]
        predicted_obs_ = self._forward_dynamic_model(obs, BATCH.action)  # [T, B, S]
        predicted_reward = self._reward_model(obs, BATCH.action)  # [T, B, 1]
        predicted_done_dist = self._done_model(obs, BATCH.action)  # [T, B, 1]
        _obs_loss = F.mse_loss(obs_, predicted_obs_)  # todo
        _reward_loss = F.mse_loss(BATCH.reward, predicted_reward)
        _done_loss = -predicted_done_dist.log_prob(BATCH.done).mean()
        wm_loss = _obs_loss + _reward_loss + _done_loss
        self._wm_oplr.optimize(wm_loss)

        obs = th.reshape(obs, (_timestep * _batchsize, -1))  # [T*B, S]
        obs_ = th.reshape(obs_, (_timestep * _batchsize, -1))  # [T*B, S]
        actions = th.reshape(BATCH.action, (_timestep * _batchsize, -1))  # [T*B, A]
        rewards = th.reshape(BATCH.reward, (_timestep * _batchsize, -1))  # [T*B, 1]
        dones = th.reshape(BATCH.done, (_timestep * _batchsize, -1))  # [T*B, 1]

        rollout_rewards = [rewards]
        rollout_dones = [dones]

        r_obs_ = obs_
        _r_obs = deepcopy(BATCH.obs_)
        r_done = (1. - dones)

        for _ in range(self._roll_out_horizon):
            r_obs = r_obs_
            _r_obs.vector.vector_0 = r_obs
            if self.is_continuous:
                action_target = self.actor.t(_r_obs)  # [T*B, A]
                if self.use_target_action_noise:
                    r_action = self.target_noised_action(action_target)  # [T*B, A]
            else:
                target_logits = self.actor.t(_r_obs)  # [T*B, A]
                target_cate_dist = td.Categorical(logits=target_logits)
                target_pi = target_cate_dist.sample()  # [T*B,]
                r_action = F.one_hot(target_pi, self.a_dim).float()  # [T*B, A]
            r_obs_ = self._forward_dynamic_model(r_obs, r_action)  # [T*B, S]
            r_reward = self._reward_model(r_obs, r_action)  # [T*B, 1]
            r_done = r_done * (1. - self._done_model(r_obs, r_action).sample())  # [T*B, 1]

            rollout_rewards.append(r_reward)  # [H+1, T*B, 1]
            rollout_dones.append(r_done)  # [H+1, T*B, 1]

        _r_obs.vector.vector_0 = obs
        q = self.critic(_r_obs, actions)  # [T*B, 1]
        _r_obs.vector.vector_0 = r_obs_
        q_target = self.critic.t(_r_obs, r_action)  # [T*B, 1]
        dc_r = rewards
        for t in range(1, self._roll_out_horizon):
            dc_r += (self.gamma ** t) * (rollout_rewards[t] * rollout_dones[t])
        dc_r += (self.gamma ** self._roll_out_horizon) * rollout_dones[self._roll_out_horizon] * q_target  # [T*B, 1]

        td_error = dc_r - q  # [T*B, 1]
        q_loss = td_error.square().mean()  # 1
        self.critic_oplr.optimize(q_loss)

        # train actor
        if self.is_continuous:
            mu = self.actor(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
        else:
            logits = self.actor(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
            logp_all = logits.log_softmax(-1)  # [T, B, A]
            gumbel_noise = td.Gumbel(0, 1).sample(logp_all.shape)  # [T, B, A]
            _pi = ((logp_all + gumbel_noise) / self.discrete_tau).softmax(-1)  # [T, B, A]
            _pi_true_one_hot = F.one_hot(_pi.argmax(-1), self.a_dim).float()  # [T, B, A]
            _pi_diff = (_pi_true_one_hot - _pi).detach()  # [T, B, A]
            mu = _pi_diff + _pi  # [T, B, A]
        q_actor = self.critic(BATCH.obs, mu, begin_mask=BATCH.begin_mask)  # [T, B, 1]
        actor_loss = -q_actor.mean()  # 1
        self.actor_oplr.optimize(actor_loss)

        return th.ones_like(BATCH.reward), {
            'LEARNING_RATE/wm_lr': self._wm_oplr.lr,
            'LEARNING_RATE/actor_lr': self.actor_oplr.lr,
            'LEARNING_RATE/critic_lr': self.critic_oplr.lr,
            'LOSS/wm_loss': wm_loss,
            'LOSS/actor_loss': actor_loss,
            'LOSS/critic_loss': q_loss,
            'Statistics/q_min': q.min(),
            'Statistics/q_mean': q.mean(),
            'Statistics/q_max': q.max()
        }
