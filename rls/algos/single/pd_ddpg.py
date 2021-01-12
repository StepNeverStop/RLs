#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from collections import namedtuple

from rls.nn.noise import (OrnsteinUhlenbeckNoisedAction,
                          ClippedNormalNoisedAction)
from rls.algos.base.off_policy import Off_Policy
from rls.utils.tf2_utils import update_target_net_weights
from rls.utils.build_networks import ACCNetwork
from rls.utils.specs import (OutputNetworkType,
                             BatchExperiences)

PD_DDPG_BatchExperiences = namedtuple('PD_DDPG_BatchExperiences', BatchExperiences._fields + ('cost',))


class PD_DDPG(Off_Policy):
    '''
    Accelerated Primal-Dual Policy Optimization for Safe Reinforcement Learning, http://arxiv.org/abs/1802.06480
    Refer to https://github.com/anita-hu/TF2-RL/blob/master/Primal-Dual_DDPG/TF2_PD_DDPG_Basic.py
    '''

    def __init__(self,
                 envspec,

                 ployak=0.995,
                 use_target_action_noise=False,
                 gaussian_noise_sigma=0.2,
                 gaussian_noise_bound=0.2,
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
        self._lambda = tf.Variable(0.0, dtype=tf.float32)
        self.cost_constraint = cost_constraint  # long tern cost <= d
        self.use_target_action_noise = use_target_action_noise
        self.gaussian_noise_sigma = gaussian_noise_sigma
        self.gaussian_noise_bound = gaussian_noise_bound

        if self.is_continuous:
            # NOTE: value_net is reward net; value_net2 is cost net.
            def _create_net(name, representation_net=None): return ACCNetwork(
                name=name,
                representation_net=representation_net,
                policy_net_type=OutputNetworkType.ACTOR_DPG,
                policy_net_kwargs=dict(output_shape=self.a_dim,
                                       network_settings=network_settings['actor_continuous']),
                value_net_type=OutputNetworkType.CRITIC_QVALUE_ONE,
                value_net_kwargs=dict(action_dim=self.a_dim,
                                      network_settings=network_settings['reward']),
                value_net2_type=OutputNetworkType.CRITIC_QVALUE_ONE,
                value_net2_kwargs=dict(action_dim=self.a_dim,
                                       network_settings=network_settings['cost'])
            )
            self.target_noised_action = ClippedNormalNoisedAction(sigma=self.gaussian_noise_sigma, noise_bound=self.gaussian_noise_bound)
            self.noised_action = OrnsteinUhlenbeckNoisedAction(sigma=0.2)
        else:
            def _create_net(name, representation_net=None): return ACCNetwork(
                name=name,
                representation_net=representation_net,
                policy_net_type=OutputNetworkType.ACTOR_DCT,
                policy_net_kwargs=dict(output_shape=self.a_dim,
                                       network_settings=network_settings['actor_discrete']),
                value_net_type=OutputNetworkType.CRITIC_QVALUE_ONE,
                value_net_kwargs=dict(action_dim=self.a_dim,
                                      network_settings=network_settings['reward']),
                value_net2_type=OutputNetworkType.CRITIC_QVALUE_ONE,
                value_net2_kwargs=dict(action_dim=self.a_dim,
                                       network_settings=network_settings['cost'])
            )
            self.gumbel_dist = tfp.distributions.Gumbel(0, 1)

        self.ac_net = _create_net('ac_net', self._representation_net)
        self._representation_target_net = self._create_representation_net('_representation_target_net')
        self.ac_target_net = _create_net('ac_target_net', self._representation_target_net)

        update_target_net_weights(self.ac_target_net.weights, self.ac_net.weights)
        self.lambda_lr = lambda_lr
        self.actor_lr, self.reward_critic_lr, self.cost_critic_lr = map(self.init_lr, [actor_lr, reward_critic_lr, cost_critic_lr])
        self.optimizer_actor, self.optimizer_reward_critic, self.optimizer_cost_critic = map(self.init_optimizer, [self.actor_lr, self.reward_critic_lr, self.cost_critic_lr])

        self._worker_params_dict.update(self.ac_net._policy_models)

        self._all_params_dict.update(self.ac_net._all_models)
        self._all_params_dict.update(optimizer_actor=self.optimizer_actor,
                                     optimizer_reward_critic=self.optimizer_reward_critic,
                                     optimizer_cost_critic=self.optimizer_cost_critic)
        self._model_post_process()

    def reset(self):
        super().reset()
        if self.is_continuous:
            self.noised_action.reset()

    def choose_action(self, obs, evaluation=False):
        mu, pi, self.cell_state = self._get_action(obs, self.cell_state)
        a = mu.numpy() if evaluation else pi.numpy()
        return a

    @tf.function
    def _get_action(self, obs, cell_state):
        with tf.device(self.device):
            output, cell_state = self.ac_net(obs, cell_state=cell_state)
            if self.is_continuous:
                mu = output
                pi = self.noised_action(mu)
            else:
                logits = output
                mu = tf.argmax(logits, axis=1)
                cate_dist = tfp.distributions.Categorical(logits=logits)
                pi = cate_dist.sample()
            return mu, pi, cell_state

    def _target_params_update(self):
        update_target_net_weights(self.ac_target_net.weights, self.ac_net.weights, self.ployak)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([
                    ['LEARNING_RATE/actor_lr', self.actor_lr(self.train_step)],
                    ['LEARNING_RATE/reward_critic_lr', self.reward_critic_lr(self.train_step)],
                    ['LEARNING_RATE/cost_critic_lr', self.cost_critic_lr(self.train_step)]
                ])
            })

    @tf.function
    def _train(self, BATCH, isw, cell_state):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                feat, _ = self._representation_net(BATCH.obs, cell_state=cell_state)
                feat_, _ = self._representation_target_net(BATCH.obs_, cell_state=cell_state)

                if self.is_continuous:
                    action_target = self.ac_target_net.policy_net(feat_)
                    if self.use_target_action_noise:
                        action_target = self.target_noised_action(action_target)
                    mu = self.ac_net.policy_net(feat)
                else:
                    target_logits = self.ac_target_net.policy_net(feat_)
                    target_cate_dist = tfp.distributions.Categorical(logits=target_logits)
                    target_pi = target_cate_dist.sample()
                    target_log_pi = target_cate_dist.log_prob(target_pi)
                    action_target = tf.one_hot(target_pi, self.a_dim, dtype=tf.float32)

                    logits = self.ac_net.policy_net(feat)
                    _pi = tf.nn.softmax(logits)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(logits, axis=-1), self.a_dim, dtype=tf.float32)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    mu = _pi_diff + _pi
                q_reward = self.ac_net.value_net(feat, BATCH.action)
                q_target = self.ac_target_net.value_net(feat_, action_target)
                dc_r = tf.stop_gradient(BATCH.reward + self.gamma * q_target * (1 - BATCH.done))
                td_error_reward = q_reward - dc_r
                reward_loss = 0.5 * tf.reduce_mean(tf.square(td_error_reward) * isw)

                q_cost = self.ac_net.value_net2(tf.stop_gradient(feat), BATCH.action)
                q_target = self.ac_target_net.value_net2(feat_, action_target)
                dc_r = tf.stop_gradient(BATCH.cost + self.gamma * q_target * (1 - BATCH.done))
                td_error_cost = q_cost - dc_r
                cost_loss = 0.5 * tf.reduce_mean(tf.square(td_error_cost) * isw)

                q_loss = reward_loss + cost_loss

                reward_actor = self.ac_net.value_net(feat, mu)
                cost_actor = self.ac_net.value_net2(feat, mu)
                actor_loss = -tf.reduce_mean(reward_actor - self._lambda * cost_actor)
            q_grads = tape.gradient(reward_loss, self.ac_net.value_net_trainable_variables)
            self.optimizer_reward_critic.apply_gradients(
                zip(q_grads, self.ac_net.value_net_trainable_variables)
            )
            q_grads = tape.gradient(cost_loss, self.ac_net.value_net2_trainable_variables)
            self.optimizer_cost_critic.apply_gradients(
                zip(q_grads, self.ac_net.value_net2_trainable_variables)
            )
            actor_grads = tape.gradient(actor_loss, self.ac_net.actor_trainable_variables)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.ac_net.actor_trainable_variables)
            )

            # update dual variable
            lambda_update = tf.reduce_mean(cost_actor - self.cost_constraint)
            self._lambda.assign_add(self.lambda_lr * lambda_update)
            self._lambda.assign(tf.maximum(self._lambda, 0.0))

            self.global_step.assign_add(1)
            return (td_error_reward + td_error_cost) / 2, dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/reward_loss', reward_loss],
                ['LOSS/cost_loss', cost_loss],
                ['LOSS/q_loss', q_loss],
                ['Statistics/q_reward_min', tf.reduce_min(q_reward)],
                ['Statistics/q_reward_mean', tf.reduce_mean(q_reward)],
                ['Statistics/q_reward_max', tf.reduce_max(q_reward)],
                ['Statistics/q_cost_min', tf.reduce_min(q_cost)],
                ['Statistics/q_cost_mean', tf.reduce_mean(q_cost)],
                ['Statistics/q_cost_max', tf.reduce_max(q_cost)],
                ['Statistics/_lambda', self._lambda],
                ['Statistics/lambda_update', lambda_update]
            ])

    def get_cost(self, exps: BatchExperiences):
        return np.abs(exps.obs_.first_vector())[:, :1]    # CartPole

    def store_data(self, exps: BatchExperiences):
        # self._running_average()
        self.data.add(PD_DDPG_BatchExperiences(*exps, self.get_cost(exps)))

    def no_op_store(self, exps: BatchExperiences)
    # self._running_average()
    self.data.add(PD_DDPG_BatchExperiences(*exps, self.get_cost(exps)))
