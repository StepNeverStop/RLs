#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rls.utils.tf2_utils import (gaussian_clip_rsample,
                                 gaussian_likelihood_sum,
                                 gaussian_entropy)
from rls.algos.base.off_policy import Off_Policy
from rls.utils.build_networks import ACNetwork
from rls.utils.indexs import OutputNetworkType


class AC(Off_Policy):
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
            self.net = ACNetwork(
                name='net',
                representation_net=self._representation_net,
                policy_net_type=OutputNetworkType.ACTOR_MU_LOGSTD,
                policy_net_kwargs=dict(output_shape=self.a_dim,
                                       condition_sigma=condition_sigma,
                                       network_settings=network_settings['actor_continuous']),
                value_net_type=OutputNetworkType.CRITIC_QVALUE_ONE,
                value_net_kwargs=dict(action_dim=self.a_dim,
                                      network_settings=network_settings['critic'])
            )
        else:
            self.net = ACNetwork(
                name='net',
                representation_net=self._representation_net,
                policy_net_type=OutputNetworkType.ACTOR_DCT,
                policy_net_kwargs=dict(output_shape=self.a_dim,
                                       network_settings=network_settings['actor_discrete']),
                value_net_type=OutputNetworkType.CRITIC_QVALUE_ONE,
                value_net_kwargs=dict(action_dim=self.a_dim,
                                      network_settings=network_settings['critic'])
            )
        self.actor_lr, self.critic_lr = map(self.init_lr, [actor_lr, critic_lr])
        self.optimizer_actor, self.optimizer_critic = map(self.init_optimizer, [self.actor_lr, self.critic_lr])

        self._worker_params_dict.update(self.net._policy_models)

        self._all_params_dict.update(self.net._all_models)
        self._all_params_dict.update(optimizer_actor=self.optimizer_actor,
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
            output, cell_state = self.net(s, visual_s, cell_state=cell_state)
            if self.is_continuous:
                mu, log_std = output
                sample_op, _ = gaussian_clip_rsample(mu, log_std)
                log_prob = gaussian_likelihood_sum(sample_op, mu, log_std)
            else:
                logits = output
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
            with tf.GradientTape(persistent=True) as tape:
                (feat, feat_), _ = self._representation_net(ss, vvss, cell_state=cell_state, need_split=True)
                if self.is_continuous:
                    mu, log_std = self.net.policy_net(feat)
                    log_prob = gaussian_likelihood_sum(a, mu, log_std)
                    entropy = gaussian_entropy(log_std)

                    next_mu, _ = self.net.policy_net(feat_)
                    max_q_next = tf.stop_gradient(self.net.value_net(feat_, next_mu))
                else:
                    logits = self.net.policy_net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    log_prob = tf.reduce_sum(tf.multiply(logp_all, a), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))

                    logits = self.net.policy_net(feat_)
                    max_a = tf.argmax(logits, axis=1)
                    max_a_one_hot = tf.one_hot(max_a, self.a_dim)
                    max_q_next = tf.stop_gradient(self.net.value_net(feat_, max_a_one_hot))
                q = self.net.value_net(feat, a)
                ratio = tf.stop_gradient(tf.exp(log_prob - old_log_prob))
                q_value = tf.stop_gradient(q)
                td_error = q - (r + self.gamma * (1 - done) * max_q_next)
                critic_loss = tf.reduce_mean(tf.square(td_error) * isw)
                actor_loss = -tf.reduce_mean(ratio * log_prob * q_value)
            critic_grads = tape.gradient(critic_loss, self.net.critic_trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.net.critic_trainable_variables)
            )
            actor_grads = tape.gradient(actor_loss, self.net.actor_trainable_variables)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.net.actor_trainable_variables)
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
