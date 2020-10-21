#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rls.nn import actor_mu_logstd as ActorCts
from rls.nn import actor_discrete as ActorDcs
from rls.nn import critic_v as Critic
from rls.utils.tf2_utils import (gaussian_clip_rsample,
                                 gaussian_likelihood_sum,
                                 gaussian_entropy)
from rls.algos.base.on_policy import make_on_policy_class


class A2C(make_on_policy_class(mode='share')):
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
            self.actor_net = ActorCts(self.feat_dim, self.a_dim, condition_sigma, network_settings['actor_continuous'])
        else:
            self.actor_net = ActorDcs(self.feat_dim, self.a_dim, network_settings['actor_discrete'])
        self.actor_tv = self.actor_net.trainable_variables
        self.critic_net = Critic(self.feat_dim, network_settings['critic'])
        self.critic_tv = self.critic_net.trainable_variables + self.other_tv
        self.actor_lr, self.critic_lr = map(self.init_lr, [actor_lr, critic_lr])
        self.optimizer_actor, self.optimizer_critic = map(self.init_optimizer, [self.actor_lr, self.critic_lr])

        self.initialize_data_buffer()

        self._worker_params_dict.update(actor=self.actor_net)
        self._residual_params_dict.update(
            critic=self.critic_net,
            optimizer_actor=self.optimizer_actor,
            optimizer_critic=self.optimizer_critic)
        self._model_post_process()

    def choose_action(self, s, visual_s, evaluation=False):
        a, self.cell_state = self._get_action(s, visual_s, self.cell_state)
        a = a.numpy()
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state):
        with tf.device(self.device):
            feat, cell_state = self.get_feature(s, visual_s, cell_state=cell_state, record_cs=True)
            if self.is_continuous:
                mu, log_std = self.actor_net(feat)
                sample_op, _ = gaussian_clip_rsample(mu, log_std)
            else:
                logits = self.actor_net(feat)
                norm_dist = tfp.distributions.Categorical(logits)
                sample_op = norm_dist.sample()
        return sample_op, cell_state

    @tf.function
    def _get_value(self, feat):
        with tf.device(self.device):
            value = self.critic_net(feat)
            return value

    def calculate_statistics(self):
        feat, self.cell_state = self.get_feature(self.data.last_s(), self.data.last_visual_s(), cell_state=self.cell_state, record_cs=True)
        init_value = np.squeeze(self._get_value(feat).numpy())
        self.data.cal_dc_r(self.gamma, init_value)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        def _train(data, crsty_loss, cell_state):
            for _ in range(self.epoch):
                actor_loss, critic_loss, entropy = self.train(data, crsty_loss, cell_state)

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
    def train(self, memories, crsty_loss, cell_state):
        s, visual_s, a, dc_r = memories
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat = self.get_feature(s, visual_s, cell_state=cell_state)
                v = self.critic_net(feat)
                td_error = dc_r - v
                critic_loss = tf.reduce_mean(tf.square(td_error)) + crsty_loss
            critic_grads = tape.gradient(critic_loss, self.critic_tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_tv)
            )
            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu, log_std = self.actor_net(feat)
                    log_act_prob = gaussian_likelihood_sum(a, mu, log_std)
                    entropy = gaussian_entropy(log_std)
                else:
                    logits = self.actor_net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    log_act_prob = tf.reduce_sum(a * logp_all, axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                v = self.critic_net(feat)
                advantage = tf.stop_gradient(dc_r - v)
                actor_loss = -(tf.reduce_mean(log_act_prob * advantage) + self.beta * entropy)
            if self.is_continuous:
                actor_grads = tape.gradient(actor_loss, self.actor_tv)
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_tv)
                )
            else:
                actor_grads = tape.gradient(actor_loss, self.actor_tv)
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_tv)
                )
            self.global_step.assign_add(1)
            return actor_loss, critic_loss, entropy

    @tf.function(experimental_relax_shapes=True)
    def train_persistent(self, memories, crsty_loss, cell_state):
        s, visual_s, a, dc_r = memories
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                feat = self.get_feature(s, visual_s, cell_state=cell_state)
                if self.is_continuous:
                    mu, log_std = self.actor_net(feat)
                    log_act_prob = gaussian_likelihood_sum(a, mu, log_std)
                    entropy = gaussian_entropy(log_std)
                else:
                    logits = self.actor_net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    log_act_prob = tf.reduce_sum(a * logp_all, axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                v = self.critic_net(feat)
                advantage = tf.stop_gradient(dc_r - v)
                td_error = dc_r - v
                critic_loss = tf.reduce_mean(tf.square(td_error)) + crsty_loss
                actor_loss = -(tf.reduce_mean(log_act_prob * advantage) + self.beta * entropy)
            critic_grads = tape.gradient(critic_loss, self.critic_tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_tv)
            )
            if self.is_continuous:
                actor_grads = tape.gradient(actor_loss, self.actor_tv)
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_tv)
                )
            else:
                actor_grads = tape.gradient(actor_loss, self.actor_tv)
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_tv)
                )
            self.global_step.assign_add(1)
            return actor_loss, critic_loss, entropy
