#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rls.nn.noise import (OrnsteinUhlenbeckActionNoise,
                          ClippedNormalActionNoise)
from rls.algos.base.off_policy import Off_Policy
from rls.utils.tf2_utils import update_target_net_weights
from rls.utils.build_networks import ADoubleCNetwork
from rls.utils.indexs import OutputNetworkType


class TD3(Off_Policy):
    '''
    Twin Delayed Deep Deterministic Policy Gradient, https://arxiv.org/abs/1802.09477
    '''

    def __init__(self,
                 envspec,

                 ployak=0.995,
                 delay_num=2,
                 noise_type='gaussian',
                 gaussian_noise_sigma=0.2,
                 gaussian_noise_bound=0.2,
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 discrete_tau=1.0,
                 network_settings={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'q': [32, 32]
                 },
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.ployak = ployak
        self.delay_num = delay_num
        self.discrete_tau = discrete_tau
        self.gaussian_noise_sigma = gaussian_noise_sigma
        self.gaussian_noise_bound = gaussian_noise_bound

        if self.is_continuous:
            def _create_net(name, representation_net=None): return ADoubleCNetwork(
                name=name,
                representation_net=representation_net,
                policy_net_type=OutputNetworkType.ACTOR_DPG,
                policy_net_kwargs=dict(output_shape=self.a_dim,
                                       network_settings=network_settings['actor_continuous']),
                value_net_type=OutputNetworkType.CRITIC_QVALUE_ONE,
                value_net_kwargs=dict(action_dim=self.a_dim,
                                      network_settings=network_settings['q'])
            )
        else:
            def _create_net(name, representation_net=None): return ADoubleCNetwork(
                name=name,
                representation_net=representation_net,
                policy_net_type=OutputNetworkType.ACTOR_DCT,
                policy_net_kwargs=dict(output_shape=self.a_dim,
                                       network_settings=network_settings['actor_discrete']),
                value_net_type=OutputNetworkType.CRITIC_QVALUE_ONE,
                value_net_kwargs=dict(action_dim=self.a_dim,
                                      network_settings=network_settings['q'])
            )
            if noise_type == 'gaussian':
                self.action_noise = ClippedNormalActionNoise(sigma=self.gaussian_noise_sigma, bound=self.gaussian_noise_bound)
            elif noise_type == 'ou':
                self.action_noise = OrnsteinUhlenbeckActionNoise(sigma=0.2)
            self.gumbel_dist = tfp.distributions.Gumbel(0, 1)

        self.ac_net = _create_net('ac_net', self._representation_net)
        self._representation_target_net = self._create_representation_net('_representation_target_net')
        self.ac_target_net = _create_net('ac_target_net', self._representation_target_net)

        update_target_net_weights(self.ac_target_net.weights, self.ac_net.weights)
        self.actor_lr, self.critic_lr = map(self.init_lr, [actor_lr, critic_lr])
        self.optimizer_actor, self.optimizer_critic = map(self.init_optimizer, [self.actor_lr, self.critic_lr])

        self._worker_params_dict.update(self.ac_net._policy_models)

        self._all_params_dict.update(self.ac_net._all_models)
        self._all_params_dict.update(optimizer_actor=self.optimizer_actor,
                                     optimizer_critic=self.optimizer_critic)
        self._model_post_process()

    def reset(self):
        super().reset()
        if self.is_continuous:
            self.action_noise.reset()

    def choose_action(self, s, visual_s, evaluation=False):
        mu, pi, self.cell_state = self._get_action(s, visual_s, self.cell_state)
        a = mu.numpy() if evaluation else pi.numpy()
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state):
        with tf.device(self.device):
            output, cell_state = self.ac_net(s, visual_s, cell_state=cell_state)
            if self.is_continuous:
                mu = output
                pi = tf.clip_by_value(mu + self.action_noise(mu.shape), -1, 1)
            else:
                logits = output
                mu = tf.argmax(logits, axis=1)
                cate_dist = tfp.distributions.Categorical(logits=tf.nn.log_softmax(logits))
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
                    ['LEARNING_RATE/critic_lr', self.critic_lr(self.train_step)]
                ]),
                'train_data_list': ['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done']
            })

    @tf.function(experimental_relax_shapes=True)
    def _train(self, memories, isw, cell_state):
        s, visual_s, a, r, s_, visual_s_, done = memories
        batch_size = tf.shape(a)[0]
        with tf.device(self.device):
            for _ in range(2):
                with tf.GradientTape(persistent=True) as tape:
                    feat, _ = self._representation_net(s, visual_s, cell_state=cell_state)
                    feat_, _ = self._representation_target_net(s_, visual_s_, cell_state=cell_state)
                    if self.is_continuous:
                        action_target = self.ac_target_net.policy_net(feat_)
                        mu = self.ac_net.policy_net(feat)
                    else:
                        target_logits = self.ac_target_net.policy_net(feat_)
                        logp_all = tf.nn.log_softmax(target_logits)
                        gumbel_noise = tf.cast(self.gumbel_dist.sample([batch_size, self.a_dim]), dtype=tf.float32)
                        _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                        _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_dim)
                        _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                        action_target = _pi_diff + _pi
                        logits = self.ac_net.policy_net(feat)
                        _pi = tf.nn.softmax(logits)
                        _pi_true_one_hot = tf.one_hot(tf.argmax(logits, axis=-1), self.a_dim, dtype=tf.float32)
                        _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                        mu = _pi_diff + _pi
                    q1, q2 = self.ac_net.get_value(feat, a)
                    q1_actor = self.ac_net.value_net(feat, mu)
                    q_target = self.ac_target_net.get_min(feat_, action_target)
                    dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                    td_error1 = q1 - dc_r
                    td_error2 = q2 - dc_r
                    q1_loss = tf.reduce_mean(tf.square(td_error1) * isw)
                    q2_loss = tf.reduce_mean(tf.square(td_error2) * isw)
                    critic_loss = 0.5 * (q1_loss + q2_loss)
                    actor_loss = -tf.reduce_mean(q1_actor)
                critic_grads = tape.gradient(critic_loss, self.ac_net.critic_trainable_variables)
                self.optimizer_critic.apply_gradients(
                    zip(critic_grads, self.ac_net.critic_trainable_variables)
                )
            actor_grads = tape.gradient(actor_loss, self.ac_net.actor_trainable_variables)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.ac_net.actor_trainable_variables)
            )
            self.global_step.assign_add(1)
            return (td_error1 + td_error2) / 2, dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/q_min', tf.reduce_min(tf.minimum(q1, q2))],
                ['Statistics/q_mean', tf.reduce_mean(tf.minimum(q1, q2))],
                ['Statistics/q_max', tf.reduce_max(tf.maximum(q1, q2))]
            ])
