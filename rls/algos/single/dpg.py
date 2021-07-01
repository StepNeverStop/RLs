#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rls.nn import ActorDPG as ActorCts
from rls.nn import ActorDct
from rls.nn import CriticQvalueOne as Critic
from rls.nn.noise import (OrnsteinUhlenbeckNoisedAction,
                          ClippedNormalNoisedAction)
from rls.algos.base.off_policy import Off_Policy
from rls.utils.build_networks import ACNetwork
from rls.utils.specs import OutputNetworkType


class DPG(Off_Policy):
    '''
    Deterministic Policy Gradient, https://hal.inria.fr/file/index/docid/938992/filename/dpg-icml2014.pdf
    '''
    # off-policy DPG

    def __init__(self,
                 envspec,

                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 use_target_action_noise=False,
                 gaussian_noise_sigma=0.2,
                 gaussian_noise_bound=0.2,
                 discrete_tau=1.0,
                 network_settings={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'q': [32, 32]
                 },
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.discrete_tau = discrete_tau
        self.use_target_action_noise = use_target_action_noise
        self.gaussian_noise_sigma = gaussian_noise_sigma
        self.gaussian_noise_bound = gaussian_noise_bound

        if self.is_continuous:
            self.target_noised_action = ClippedNormalNoisedAction(sigma=self.gaussian_noise_sigma, noise_bound=self.gaussian_noise_bound)
            self.noised_action = OrnsteinUhlenbeckNoisedAction(sigma=0.2)
            self.net = ACNetwork(
                name='net',
                representation_net=self._representation_net,
                policy_net_type=OutputNetworkType.ACTOR_DPG,
                policy_net_kwargs=dict(output_shape=self.a_dim,
                                       network_settings=network_settings['actor_continuous']),
                value_net_type=OutputNetworkType.CRITIC_QVALUE_ONE,
                value_net_kwargs=dict(action_dim=self.a_dim,
                                      network_settings=network_settings['q'])
            )
        else:
            self.gumbel_dist = tfp.distributions.Gumbel(0, 1)
            self.net = ACNetwork(
                name='net',
                representation_net=self._representation_net,
                policy_net_type=OutputNetworkType.ACTOR_DCT,
                policy_net_kwargs=dict(output_shape=self.a_dim,
                                       network_settings=network_settings['actor_discrete']),
                value_net_type=OutputNetworkType.CRITIC_QVALUE_ONE,
                value_net_kwargs=dict(action_dim=self.a_dim,
                                      network_settings=network_settings['q'])
            )

        self.actor_lr, self.critic_lr = map(self.init_lr, [actor_lr, critic_lr])
        self.optimizer_actor, self.optimizer_critic = map(self.init_optimizer, [self.actor_lr, self.critic_lr])

        self._worker_params_dict.update(self.net._policy_models)

        self._all_params_dict.update(self.net._all_models)
        self._all_params_dict.update(optimizer_actor=self.optimizer_actor,
                                     optimizer_critic=self.optimizer_critic)
        self._model_post_process()
        self.initialize_data_buffer()

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
            output, cell_state = self.net(obs, cell_state=cell_state)
            if self.is_continuous:
                mu = output
                pi = self.noised_action(mu)
            else:
                logits = output
                mu = tf.argmax(logits, axis=1)
                cate_dist = tfp.distributions.Categorical(logits=logits)
                pi = cate_dist.sample()
            return mu, pi, cell_state

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([
                    ['LEARNING_RATE/actor_lr', self.actor_lr(self.train_step)],
                    ['LEARNING_RATE/critic_lr', self.critic_lr(self.train_step)]
                ]),
                'use_stack': True
            })

    @tf.function
    def _train(self, BATCH, isw, cell_state):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                (feat, feat_), _ = self._representation_net(BATCH.obs, cell_state=cell_state, need_split=True)
                if self.is_continuous:
                    action_target = self.ac_target_net.policy_net(feat_)
                    if self.use_target_action_noise:
                        action_target = self.target_noised_action(action_target)
                    mu = self.net.policy_net(feat)
                else:
                    target_logits = self.net.policy_net(feat_)
                    target_cate_dist = tfp.distributions.Categorical(logits=target_logits)
                    target_pi = target_cate_dist.sample()
                    target_log_pi = target_cate_dist.log_prob(target_pi)
                    action_target = tf.one_hot(target_pi, self.a_dim, dtype=tf.float32)

                    logits = self.net.policy_net(feat)
                    _pi = tf.nn.softmax(logits)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(logits, axis=-1), self.a_dim, dtype=tf.float32)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    mu = _pi_diff + _pi
                q_target = self.net.value_net(feat_, action_target)
                dc_r = tf.stop_gradient(BATCH.reward + self.gamma * q_target * (1 - BATCH.done))
                q = self.net.value_net(feat, BATCH.action)
                td_error = dc_r - q
                q_loss = 0.5 * tf.reduce_mean(tf.square(td_error) * isw)
                q_actor = self.net.value_net(feat, mu)
                actor_loss = -tf.reduce_mean(q_actor)
            q_grads = tape.gradient(q_loss, self.net.critic_trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(q_grads, self.net.critic_trainable_variables)
            )
            actor_grads = tape.gradient(actor_loss, self.net.actor_trainable_variables)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.net.actor_trainable_variables)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', q_loss],
                ['Statistics/q_min', tf.reduce_min(q)],
                ['Statistics/q_mean', tf.reduce_mean(q)],
                ['Statistics/q_max', tf.reduce_max(q)]
            ])
