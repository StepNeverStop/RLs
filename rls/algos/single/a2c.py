#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rls.utils.tf2_utils import (gaussian_clip_rsample,
                                 gaussian_likelihood_sum,
                                 gaussian_entropy)
from rls.algos.base.on_policy import On_Policy
from rls.utils.build_networks import ACNetwork
from rls.utils.specs import OutputNetworkType


class A2C(On_Policy):
    def __init__(self,
                 envspec,

                 epoch=5,
                 beta=1.0e-3,
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 condition_sigma: bool = False,
                 network_settings={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'critic': [32, 32]
                 },
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.beta = beta
        self.epoch = epoch

        if self.is_continuous:
            self.net = ACNetwork(
                name='net',
                representation_net=self._representation_net,
                policy_net_type=OutputNetworkType.ACTOR_MU_LOGSTD,
                policy_net_kwargs=dict(output_shape=self.a_dim,
                                       condition_sigma=condition_sigma,
                                       network_settings=network_settings['actor_continuous']),
                value_net_type=OutputNetworkType.CRITIC_VALUE,
                value_net_kwargs=dict(network_settings=network_settings['critic'])
            )
        else:
            self.net = ACNetwork(
                name='net',
                representation_net=self._representation_net,
                policy_net_type=OutputNetworkType.ACTOR_DCT,
                policy_net_kwargs=dict(output_shape=self.a_dim,
                                       network_settings=network_settings['actor_discrete']),
                value_net_type=OutputNetworkType.CRITIC_VALUE,
                value_net_kwargs=dict(network_settings=network_settings['critic'])
            )

        self.actor_lr, self.critic_lr = map(self.init_lr, [actor_lr, critic_lr])
        self.optimizer_actor, self.optimizer_critic = map(self.init_optimizer, [self.actor_lr, self.critic_lr])

        self.initialize_data_buffer()

        self._worker_params_dict.update(self.net._policy_models)

        self._all_params_dict.update(self.net._all_models)
        self._all_params_dict.update(optimizer_actor=self.optimizer_actor,
                                     optimizer_critic=self.optimizer_critic)
        self._model_post_process()

    def choose_action(self, s, visual_s, evaluation=False):
        a, self.next_cell_state = self._get_action(s, visual_s, self.cell_state)
        a = a.numpy()
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state):
        with tf.device(self.device):
            output, cell_state = self.net(s, visual_s, cell_state=cell_state)
            if self.is_continuous:
                mu, log_std = output
                sample_op, _ = gaussian_clip_rsample(mu, log_std)
            else:
                logits = output
                norm_dist = tfp.distributions.Categorical(logits=tf.nn.log_softmax(logits))
                sample_op = norm_dist.sample()
        return sample_op, cell_state

    @tf.function
    def _get_value(self, s, visual_s, cell_state):
        with tf.device(self.device):
            feat, cell_state = self._representation_net(s, visual_s, cell_state=cell_state)
            value = self.net.value_net(feat)
            return value, cell_state

    def calculate_statistics(self):
        init_value, self.cell_state = self._get_value(self.data.last_s(), self.data.last_visual_s(), cell_state=self.cell_state)
        init_value = np.squeeze(init_value.numpy())
        self.data.cal_dc_r(self.gamma, init_value)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        def _train(data):
            for _ in range(self.epoch):
                actor_loss, critic_loss, entropy = self.train(data)

            summaries = dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/entropy', entropy],
            ])
            return summaries

        self._learn(function_dict={
            'calculate_statistics': self.calculate_statistics,
            'train_function': _train,
            'train_data_list': ['s', 'visual_s', 'a', 'discounted_reward'],
            'summary_dict': dict([
                ['LEARNING_RATE/actor_lr', self.actor_lr(self.train_step)],
                ['LEARNING_RATE/critic_lr', self.critic_lr(self.train_step)]
            ])
        })

    @tf.function(experimental_relax_shapes=True)
    def train(self, memories):
        s, visual_s, a, dc_r, cell_state = memories
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                feat, _ = self._representation_net(s, visual_s, cell_state=cell_state)
                if self.is_continuous:
                    mu, log_std = self.net.policy_net(feat)
                    log_act_prob = gaussian_likelihood_sum(a, mu, log_std)
                    entropy = gaussian_entropy(log_std)
                else:
                    logits = self.net.policy_net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    log_act_prob = tf.reduce_sum(a * logp_all, axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                v = self.net.value_net(feat)
                advantage = tf.stop_gradient(dc_r - v)
                td_error = dc_r - v
                critic_loss = tf.reduce_mean(tf.square(td_error))
                actor_loss = -(tf.reduce_mean(log_act_prob * advantage) + self.beta * entropy)
            critic_grads = tape.gradient(critic_loss, self.net.critic_trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.net.critic_trainable_variables)
            )
            if self.is_continuous:
                actor_grads = tape.gradient(actor_loss, self.net.actor_trainable_variables)
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.net.actor_trainable_variables)
                )
            else:
                actor_grads = tape.gradient(actor_loss, self.net.actor_trainable_variables)
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.net.actor_trainable_variables)
                )
            self.global_step.assign_add(1)
            return actor_loss, critic_loss, entropy
