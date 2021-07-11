#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from torch import distributions as td
from dataclasses import dataclass

from rls.nn.noised_actions import ClippedNormalNoisedAction
from rls.algos.base.off_policy import Off_Policy
from rls.utils.torch_utils import (sync_params_pairs,
                                   q_target_func)
from rls.common.specs import BatchExperiences
from rls.nn.models import (CriticQvalueOne,
                           ActorDct,
                           ActorDPG)
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


@dataclass(eq=False)
class PD_DDPG_BatchExperiences(BatchExperiences):
    cost: np.ndarray


class PD_DDPG(Off_Policy):
    '''
    Accelerated Primal-Dual Policy Optimization for Safe Reinforcement Learning, http://arxiv.org/abs/1802.06480
    Refer to https://github.com/anita-hu/TF2-RL/blob/master/Primal-Dual_DDPG/TF2_PD_DDPG_Basic.py
    '''

    def __init__(self,
                 envspec,

                 ployak=0.995,
                 use_target_action_noise=False,
                 noise_action='ou',
                 noise_params={
                     'sigma': 0.2
                 },
                 actor_lr=5.0e-4,
                 reward_critic_lr=1.0e-3,
                 cost_critic_lr=1.0e-3,
                 lambda_lr=5.0e-4,
                 discrete_tau=1.0,
                 cost_constraint=1.0,
                 network_settings={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'reward': [32, 32],
                     'cost': [32, 32]
                 },
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.ployak = ployak
        self.discrete_tau = discrete_tau
        self._lambda = t.tensor(0.)
        self.cost_constraint = cost_constraint  # long tern cost <= d
        self.use_target_action_noise = use_target_action_noise
        self.gaussian_noise_sigma = gaussian_noise_sigma
        self.gaussian_noise_bound = gaussian_noise_bound

        if self.is_continuous:
            # NOTE: value_net is reward net; value_net2 is cost net.
            self.actor = ActorDPG(self.rep_net.h_dim,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_continuous'])
            self.target_noised_action = ClippedNormalNoisedAction(sigma=0.2, noise_bound=0.2)
            self.noised_action = Noise_action_REGISTER[noise_action](**noise_params)
        else:
            self.actor = ActorDct(self.rep_net.h_dim,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_discrete'])
            self.gumbel_dist = td.gumbel.Gumbel(0, 1)

        self.critic_reward = CriticQvalueOne(self.rep_net.h_dim,
                                             action_dim=self.a_dim,
                                             network_settings=network_settings['reward'])
        self.critic_cost = CriticQvalueOne(self.rep_net.h_dim,
                                           action_dim=self.a_dim,
                                           network_settings=network_settings['cost'])

        self._target_rep_net = deepcopy(self.rep_net)
        self._target_rep_net.eval()
        self.actor_target = deepcopy(self.actor)
        self.actor_target.eval()
        self.critic_reward_target = deepcopy(self.critic_reward)
        self.critic_reward_target.eval()
        self.critic_cost_target = deepcopy(self.critic_cost)
        self.critic_cost_target.eval()

        self._pairs = [(self._target_rep_net, self.rep_net),
                       (self.actor_target, self.actor),
                       (self.critic_reward_target, self.critic_reward),
                       (self.critic_cost_target, self.critic_cost)]
        sync_params_pairs(self._pairs)

        self.lambda_lr = lambda_lr
        self.actor_oplr = OPLR(self.actor, actor_lr)
        self.reward_critic_oplr = OPLR([self.critic_reward, self.rep_net], reward_critic_lr)
        self.cost_critic_oplr = OPLR(self.critic_cost, cost_critic_lr)

        self._worker_modules.update(rep_net=self.rep_net,
                                    actor=self.actor)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(critic_reward=self.critic_reward,
                                     critic_cost=self.critic_reward,
                                     actor_oplr=self.actor_oplr,
                                     reward_critic_oplr=self.reward_critic_oplr,
                                     cost_critic_oplr=self.cost_critic_oplr)
        self.initialize_data_buffer()

    def reset(self):
        super().reset()
        if self.is_continuous:
            self.noised_action.reset()

    @iTensor_oNumpy
    def __call__(self, obs, evaluation=False):
        feat, self.cell_state = self.rep_net(obs, cell_state=self.cell_state)
        output = self.actor(feat)
        if self.is_continuous:
            mu = output
            pi = self.noised_action(mu)
        else:
            logits = output
            mu = logits.argmax(1)
            cate_dist = td.categorical.Categorical(logits=logits)
            pi = cate_dist.sample()
        return mu if evaluation else pi

    def _target_params_update(self):
        sync_params_pairs(self._pairs, self.ployak)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([
                    ['LEARNING_RATE/actor_lr', self.actor_oplr.lr],
                    ['LEARNING_RATE/reward_critic_lr', self.reward_critic_oplr.lr],
                    ['LEARNING_RATE/cost_critic_lr', self.cost_critic_oplr.lr]
                ])
            })

    @iTensor_oNumpy
    def _train(self, BATCH, isw, cell_states):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
        feat_, _ = self._target_rep_net(BATCH.obs_, cell_state=cell_states['obs_'])

        if self.is_continuous:
            action_target = self.actor_target(feat_)
            if self.use_target_action_noise:
                action_target = self.target_noised_action(action_target)
            mu = self.actor(feat)
        else:
            target_logits = self.actor_target(feat_)
            target_cate_dist = td.categorical.Categorical(logits=target_logits)
            target_pi = target_cate_dist.sample()
            target_log_pi = target_cate_dist.log_prob(target_pi)
            action_target = t.nn.functional.one_hot(target_pi, self.a_dim).float()

            logits = self.actor(feat)
            _pi = logits.softmax(-1)
            _pi_true_one_hot = t.nn.functional.one_hot(logits.argmax(-1), self.a_dim).float()
            _pi_diff = (_pi_true_one_hot - _pi).detach()
            mu = _pi_diff + _pi
        q_reward = self.critic_reward(feat, BATCH.action)
        q_target = self.critic_reward_target(feat_, action_target)
        dc_r = q_target_func(BATCH.reward,
                             self.gamma,
                             BATCH.done,
                             q_target)
        td_error_reward = q_reward - dc_r
        reward_loss = 0.5 * (td_error_reward.square() * isw).mean()

        q_cost = self.critic_cost(feat.detach(), BATCH.action)
        q_target = self.critic_cost_target(feat_, action_target)
        dc_r = q_target_func(BATCH.cost,
                             self.gamma,
                             BATCH.done,
                             q_target)
        td_error_cost = q_cost - dc_r
        cost_loss = 0.5 * (td_error_cost.square() * isw).mean()

        q_loss = reward_loss + cost_loss

        reward_actor = self.critic_reward(feat, mu)
        cost_actor = self.critic_cost(feat, mu)
        actor_loss = -(reward_actor - self._lambda * cost_actor).mean()

        self.reward_critic_oplr.step(reward_loss)
        self.cost_critic_oplr.step(cost_loss)
        self.actor_oplr.step(actor_loss)

        # update dual variable
        lambda_update = (cost_actor - self.cost_constraint).mean()
        self._lambda.add_(self.lambda_lr * lambda_update)
        self._lambda = t.maximum(self._lambda, t.zeros_like(self._lambda))

        self.global_step.add_(1)
        return (td_error_reward + td_error_cost) / 2, dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/reward_loss', reward_loss],
            ['LOSS/cost_loss', cost_loss],
            ['LOSS/q_loss', q_loss],
            ['Statistics/q_reward_min', q_reward.min()],
            ['Statistics/q_reward_mean', q_reward.mean()],
            ['Statistics/q_reward_max', q_reward.max()],
            ['Statistics/q_cost_min', q_cost.min()],
            ['Statistics/q_cost_mean', q_cost.mean()],
            ['Statistics/q_cost_max', q_cost.max()],
            ['Statistics/_lambda', self._lambda],
            ['Statistics/lambda_update', lambda_update]
        ])

    def get_cost(self, exps: BatchExperiences):
        return np.abs(exps.obs_.first_vector())[:, :1]    # CartPole

    def store_data(self, exps: BatchExperiences):
        # self._running_average()
        self.data.add(PD_DDPG_BatchExperiences(*exps.astuple(), self.get_cost(exps)))

    def no_op_store(self, exps: BatchExperiences):
        # self._running_average()
        self.data.add(PD_DDPG_BatchExperiences(*exps.astuple(), self.get_cost(exps)))
