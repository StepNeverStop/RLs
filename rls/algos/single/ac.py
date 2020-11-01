#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rls.nn import actor_mu_logstd as ActorCts
from rls.nn import actor_discrete as ActorDcs
from rls.nn import critic_q_one as Critic
from rls.utils.tf2_utils import (gaussian_clip_rsample,
                                 gaussian_likelihood_sum,
                                 gaussian_entropy)
from rls.algos.base.off_policy import make_off_policy_class


class AC(make_off_policy_class(mode='share')):
    # off-policy actor-critic
    def __init__(self,
                 envspec,

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

        if self.is_continuous:
            self.actor_net = ActorCts(self.feat_dim, self.a_dim, condition_sigma, network_settings['actor_continuous'])
        else:
            self.actor_net = ActorDcs(self.feat_dim, self.a_dim, network_settings['actor_discrete'])
        self.actor_tv = self.actor_net.trainable_variables
        self.critic_net = Critic(self.feat_dim, self.a_dim, network_settings['critic'])
        self.critic_tv = self.critic_net.trainable_variables + self.other_tv
        self.actor_lr, self.critic_lr = map(self.init_lr, [actor_lr, critic_lr])
        self.optimizer_actor, self.optimizer_critic = map(self.init_optimizer, [self.actor_lr, self.critic_lr])

        self._worker_params_dict.update(actor=self.actor_net)
        self._residual_params_dict.update(
            critic=self.critic_net,
            optimizer_actor=self.optimizer_actor,
            optimizer_critic=self.optimizer_critic)
        self._model_post_process()

    def choose_action(self, s, visual_s, evaluation=False):
        a, _lp, self.cell_state = self._get_action(s, visual_s, self.cell_state)
        a = a.numpy()
        self._log_prob = _lp.numpy()
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state):
        with tf.device(self.device):
            feat, cell_state = self.get_feature(s, visual_s, cell_state=cell_state, record_cs=True)
            if self.is_continuous:
                mu, log_std = self.actor_net(feat)
                sample_op, _ = gaussian_clip_rsample(mu, log_std)
                log_prob = gaussian_likelihood_sum(sample_op, mu, log_std)
            else:
                logits = self.actor_net(feat)
                norm_dist = tfp.distributions.Categorical(logits=tf.nn.log_softmax(logits))
                sample_op = norm_dist.sample()
                log_prob = norm_dist.log_prob(sample_op)
        return sample_op, log_prob, cell_state

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "store_data need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store_data need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store_data need done type is np.ndarray"
        self._running_average(s)
        old_log_prob = self._log_prob
        self.data.add(s, visual_s, a, r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis], old_log_prob)

    def no_op_store(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "store_data need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store_data need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store_data need done type is np.ndarray"
        self._running_average(s)
        old_log_prob = np.ones_like(r)
        self.data.add(s, visual_s, a, r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis], old_log_prob[:, np.newaxis])

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([
                    ['LEARNING_RATE/actor_lr', self.actor_lr(self.train_step)],
                    ['LEARNING_RATE/critic_lr', self.critic_lr(self.train_step)]
                ]),
                'sample_data_list': ['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done', 'old_log_prob'],
                'train_data_list': ['ss', 'vvss', 'a', 'r', 'done', 'old_log_prob']
            })

    @tf.function(experimental_relax_shapes=True)
    def _train(self, memories, isw, cell_state):
        ss, vvss, a, r, done, old_log_prob = memories
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                if self.is_continuous:
                    next_mu, _ = self.actor_net(feat_)
                    max_q_next = tf.stop_gradient(self.critic_net(feat_, next_mu))
                else:
                    logits = self.actor_net(feat_)
                    max_a = tf.argmax(logits, axis=1)
                    max_a_one_hot = tf.one_hot(max_a, self.a_dim, dtype=tf.float32)
                    max_q_next = tf.stop_gradient(self.critic_net(feat_, max_a_one_hot))
                q = self.critic_net(feat, a)
                td_error = q - (r + self.gamma * (1 - done) * max_q_next)
                critic_loss = tf.reduce_mean(tf.square(td_error) * isw)
            critic_grads = tape.gradient(critic_loss, self.critic_tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_tv)
            )
            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu, log_std = self.actor_net(feat)
                    log_prob = gaussian_likelihood_sum(a, mu, log_std)
                    entropy = gaussian_entropy(log_std)
                else:
                    logits = self.actor_net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    log_prob = tf.reduce_sum(tf.multiply(logp_all, a), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                q = self.critic_net(feat, a)
                ratio = tf.stop_gradient(tf.exp(log_prob - old_log_prob))
                q_value = tf.stop_gradient(q)
                actor_loss = -tf.reduce_mean(ratio * log_prob * q_value)
            actor_grads = tape.gradient(actor_loss, self.actor_tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_tv)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/q_max', tf.reduce_max(q)],
                ['Statistics/q_min', tf.reduce_min(q)],
                ['Statistics/q_mean', tf.reduce_mean(q)],
                ['Statistics/ratio', tf.reduce_mean(ratio)],
                ['Statistics/entropy', entropy]
            ])

    @tf.function(experimental_relax_shapes=True)
    def train_persistent(self, memories, isw, cell_state):
        ss, vvss, a, r, done, old_log_prob = memories
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                if self.is_continuous:
                    next_mu, _ = self.actor_net(feat_)
                    max_q_next = tf.stop_gradient(self.critic_net(feat_, next_mu))
                    mu, log_std = self.actor_net(feat)
                    log_prob = gaussian_likelihood_sum(a, mu, log_std)
                    entropy = gaussian_entropy(log_std)
                else:
                    logits = self.actor_net(feat_)
                    max_a = tf.argmax(logits, axis=1)
                    max_a_one_hot = tf.one_hot(max_a, self.a_dim)
                    max_q_next = tf.stop_gradient(self.critic_net(feat_, max_a_one_hot))
                    logits = self.actor_net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    log_prob = tf.reduce_sum(tf.multiply(logp_all, a), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                q = self.critic_net(feat, a)
                ratio = tf.stop_gradient(tf.exp(log_prob - old_log_prob))
                q_value = tf.stop_gradient(q)
                td_error = q - (r + self.gamma * (1 - done) * max_q_next)
                critic_loss = tf.reduce_mean(tf.square(td_error) * isw)
                actor_loss = -tf.reduce_mean(ratio * log_prob * q_value)
            critic_grads = tape.gradient(critic_loss, self.critic_tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_tv)
            )
            actor_grads = tape.gradient(actor_loss, self.actor_tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_tv)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/q_max', tf.reduce_max(q)],
                ['Statistics/q_min', tf.reduce_min(q)],
                ['Statistics/q_mean', tf.reduce_mean(q)],
                ['Statistics/ratio', tf.reduce_mean(ratio)],
                ['Statistics/entropy', entropy]
            ])
