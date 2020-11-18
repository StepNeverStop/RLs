#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rls.algos.base.off_policy import Off_Policy
from rls.utils.tf2_utils import (clip_nn_log_std,
                                 squash_rsample,
                                 gaussian_entropy,
                                 update_target_net_weights)
from rls.utils.sundry_utils import LinearAnnealing
from rls.utils.build_networks import (ValueNetwork,
                                      DoubleValueNetwork)
from rls.utils.indexs import OutputNetworkType


class SAC_V(Off_Policy):
    """
        Soft Actor Critic with Value neural network. https://arxiv.org/abs/1812.05905
        Soft Actor-Critic for Discrete Action Settings. https://arxiv.org/abs/1910.07207
    """

    def __init__(self,
                 envspec,

                 alpha=0.2,
                 annealing=True,
                 last_alpha=0.01,
                 ployak=0.995,
                 use_gumbel=True,
                 discrete_tau=1.0,
                 log_std_bound=[-20, 2],
                 network_settings={
                     'actor_continuous': {
                         'share': [128, 128],
                         'mu': [64],
                         'log_std': [64]
                     },
                     'actor_discrete': [64, 32],
                     'q': [128, 128],
                     'v': [128, 128]
                 },
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 alpha_lr=5.0e-4,
                 auto_adaption=True,
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.ployak = ployak
        self.use_gumbel = use_gumbel
        self.discrete_tau = discrete_tau
        self.log_std_min, self.log_std_max = log_std_bound[:]
        self.auto_adaption = auto_adaption
        self.annealing = annealing

        if self.auto_adaption:
            self.log_alpha = tf.Variable(initial_value=0.0, name='log_alpha', dtype=tf.float32, trainable=True)
        else:
            self.log_alpha = tf.Variable(initial_value=tf.math.log(alpha), name='log_alpha', dtype=tf.float32, trainable=False)
            if self.annealing:
                self.alpha_annealing = LinearAnnealing(alpha, last_alpha, 1e6)

        def _create_net(name, representation_net=None): return ValueNetwork(
            name=name,
            representation_net=representation_net,
            value_net_type=OutputNetworkType.CRITIC_VALUE,
            value_net_kwargs=dict(network_settings=network_settings['v'])
        )
        self.v_net = _create_net('v_net', self._representation_net)
        self._representation_target_net = self._create_representation_net('_representation_target_net')
        self.v_target_net = _create_net('v_target_net', self._representation_target_net)

        if self.is_continuous:
            self.actor_net = ValueNetwork(
                name='actor_net',
                value_net_type=OutputNetworkType.ACTOR_CTS,
                value_net_kwargs=dict(vector_dim=self._representation_net.h_dim,
                                      output_shape=self.a_dim,
                                      network_settings=network_settings['actor_continuous'])
            )
        else:
            self.actor_net = ValueNetwork(
                name='actor_net',
                value_net_type=OutputNetworkType.ACTOR_DCT,
                value_net_kwargs=dict(vector_dim=self._representation_net.h_dim,
                                      output_shape=self.a_dim,
                                      network_settings=network_settings['actor_discrete'])
            )
            if self.use_gumbel:
                self.gumbel_dist = tfp.distributions.Gumbel(0, 1)

        # entropy = -log(1/|A|) = log |A|
        self.target_entropy = 0.98 * (-self.a_dim if self.is_continuous else np.log(self.a_dim))

        if self.is_continuous or self.use_gumbel:
            self.q_net = DoubleValueNetwork(
                name='q_net',
                value_net_type=OutputNetworkType.CRITIC_QVALUE_ONE,
                value_net_kwargs=dict(vector_dim=self._representation_net.h_dim,
                                      action_dim=self.a_dim,
                                      network_settings=network_settings['q'])
            )
        else:
            self.q_net = DoubleValueNetwork(
                name='q_net',
                value_net_type=OutputNetworkType.CRITIC_QVALUE_ALL,
                value_net_kwargs=dict(vector_dim=self._representation_net.h_dim,
                                      output_shape=self.a_dim,
                                      network_settings=network_settings['q'])
            )

        update_target_net_weights(self.v_target_net.weights, self.v_net.weights)
        self.actor_lr, self.critic_lr, self.alpha_lr = map(self.init_lr, [actor_lr, critic_lr, alpha_lr])
        self.optimizer_actor, self.optimizer_critic, self.optimizer_alpha = map(self.init_optimizer, [self.actor_lr, self.critic_lr, self.alpha_lr])

        self._worker_params_dict.update(self._representation_net._all_models)
        self._worker_params_dict.update(self.actor_net._policy_models)

        self._all_params_dict.update(self.actor_net._all_models)
        self._all_params_dict.update(self.v_net._all_models)
        self._all_params_dict.update(self.q_net._all_models)
        self._all_params_dict.update(log_alpha=self.log_alpha,
                                          optimizer_actor=self.optimizer_actor,
                                          optimizer_critic=self.optimizer_critic,
                                          optimizer_alpha=self.optimizer_alpha)
        self._model_post_process()

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def choose_action(self, s, visual_s, evaluation=False):
        mu, pi, self.cell_state = self._get_action(s, visual_s, self.cell_state)
        a = mu.numpy() if evaluation else pi.numpy()
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state):
        with tf.device(self.device):
            feat, cell_state = self._representation_net(s, visual_s, cell_state=cell_state)
            if self.is_continuous:
                mu, log_std = self.actor_net.value_net(feat)
                log_std = clip_nn_log_std(log_std, self.log_std_min, self.log_std_max)
                pi, _ = squash_rsample(mu, log_std)
                mu = tf.tanh(mu)    # squash mu
            else:
                logits = self.actor_net.value_net(feat)
                mu = tf.argmax(logits, axis=1)
                cate_dist = tfp.distributions.Categorical(logits=tf.nn.log_softmax(logits))
                pi = cate_dist.sample()
            return mu, pi, cell_state

    def _target_params_update(self):
        update_target_net_weights(self.v_target_net.weights, self.v_net.weights, self.ployak)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([
                    ['LEARNING_RATE/actor_lr', self.actor_lr(self.train_step)],
                    ['LEARNING_RATE/critic_lr', self.critic_lr(self.train_step)],
                    ['LEARNING_RATE/alpha_lr', self.alpha_lr(self.train_step)]
                ]),
                'train_data_list': ['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done'],
            })

    def _train(self, memories, isw, cell_state):
        if self.is_continuous or self.use_gumbel:
            td_error, summaries = self.train_continuous(memories, isw, cell_state)
        else:
            td_error, summaries = self.train_discrete(memories, isw, cell_state)
        if self.annealing and not self.auto_adaption:
            self.log_alpha.assign(tf.math.log(tf.cast(self.alpha_annealing(self.global_step.numpy()), tf.float32)))
        return td_error, summaries

    @tf.function(experimental_relax_shapes=True)
    def train_continuous(self, memories, isw, cell_state):
        s, visual_s, a, r, s_, visual_s_, done = memories
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                feat, _ = self._representation_net(s, visual_s, cell_state=cell_state)
                v = self.v_net.value_net(feat)
                v_target, _ = self.v_target_net(s_, visual_s_, cell_state=cell_state)

                if self.is_continuous:
                    mu, log_std = self.actor_net.value_net(feat)
                    log_std = clip_nn_log_std(log_std, self.log_std_min, self.log_std_max)
                    pi, log_pi = squash_rsample(mu, log_std)
                    entropy = gaussian_entropy(log_std)
                else:
                    logits = self.actor_net.value_net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    gumbel_noise = tf.cast(self.gumbel_dist.sample(a.shape), dtype=tf.float32)
                    _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_dim)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    pi = _pi_diff + _pi
                    log_pi = tf.reduce_sum(tf.multiply(logp_all, pi), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                q1, q2 = self.q_net.get_value(feat, a)
                q1_pi, q2_pi = self.q_net.get_value(feat, pi)
                dc_r = tf.stop_gradient(r + self.gamma * v_target * (1 - done))
                v_from_q_stop = tf.stop_gradient(tf.minimum(q1_pi, q2_pi) - self.alpha * log_pi)
                td_v = v - v_from_q_stop
                td_error1 = q1 - dc_r
                td_error2 = q2 - dc_r
                q1_loss = tf.reduce_mean(tf.square(td_error1) * isw)
                q2_loss = tf.reduce_mean(tf.square(td_error2) * isw)
                v_loss_stop = tf.reduce_mean(tf.square(td_v) * isw)
                critic_loss = 0.5 * q1_loss + 0.5 * q2_loss + 0.5 * v_loss_stop
                actor_loss = -tf.reduce_mean(q1_pi - self.alpha * log_pi)
                if self.auto_adaption:
                    alpha_loss = -tf.reduce_mean(self.alpha * tf.stop_gradient(log_pi + self.target_entropy))
            actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.trainable_variables)
            )
            critic_grads = tape.gradient(critic_loss, self.q_net.trainable_variables + self.v_net.trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.q_net.trainable_variables + self.v_net.trainable_variables)
            )
            if self.auto_adaption:
                alpha_grad = tape.gradient(alpha_loss, self.log_alpha)
                self.optimizer_alpha.apply_gradients(
                    [(alpha_grad, self.log_alpha)]
                )
            self.global_step.assign_add(1)
            summaries = dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/q1_loss', q1_loss],
                ['LOSS/q2_loss', q2_loss],
                ['LOSS/v_loss', v_loss_stop],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/log_alpha', self.log_alpha],
                ['Statistics/alpha', self.alpha],
                ['Statistics/entropy', entropy],
                ['Statistics/q_min', tf.reduce_min(tf.minimum(q1, q2))],
                ['Statistics/q_mean', tf.reduce_mean(tf.minimum(q1, q2))],
                ['Statistics/q_max', tf.reduce_max(tf.maximum(q1, q2))],
                ['Statistics/v_mean', tf.reduce_mean(v)]
            ])
            if self.auto_adaption:
                summaries.update({
                    'LOSS/alpha_loss': alpha_loss
                })
            return (td_error1 + td_error2) / 2, summaries

    @tf.function(experimental_relax_shapes=True)
    def train_discrete(self, memories, isw, cell_state):
        s, visual_s, a, r, s_, visual_s_, done = memories
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                feat, _ = self._representation_net(s, visual_s, cell_state=cell_state)
                v = self.v_net.value_net(feat)  # [B, 1]
                v_target, _ = self.v_target_net(s_, visual_s_, cell_state=cell_state)  # [B, 1]

                q1_all, q2_all = self.q_net.get_value(feat)   # [B, A]
                def q_function(x): return tf.reduce_sum(x * a, axis=-1, keepdims=True)  # [B, 1]
                q1 = q_function(q1_all)
                q2 = q_function(q2_all)
                logits = self.actor_net.value_net(feat)  # [B, A]
                logp_all = tf.nn.log_softmax(logits)  # [B, A]

                entropy = -tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True)    # [B, 1]
                q_all = self.q_net.get_min(feat)   # [B, A]
                actor_loss = -tf.reduce_mean(
                    tf.reduce_sum((q_all - self.alpha * logp_all) * tf.exp(logp_all))  # [B, A] => [B,]
                )

                dc_r = tf.stop_gradient(r + self.gamma * v_target * (1 - done))
                td_v = v - tf.stop_gradient(tf.minimum(
                    tf.reduce_sum(tf.exp(logp_all) * q1_all, axis=-1),
                    tf.reduce_sum(tf.exp(logp_all) * q2_all, axis=-1)
                ))
                td_error1 = q1 - dc_r
                td_error2 = q2 - dc_r
                q1_loss = tf.reduce_mean(tf.square(td_error1) * isw)
                q2_loss = tf.reduce_mean(tf.square(td_error2) * isw)
                v_loss_stop = tf.reduce_mean(tf.square(td_v) * isw)
                critic_loss = 0.5 * q1_loss + 0.5 * q2_loss + 0.5 * v_loss_stop

                if self.auto_adaption:
                    corr = tf.stop_gradient(self.target_entropy - entropy)
                    # corr = tf.stop_gradient(tf.reduce_sum((logp_all - self.a_dim) * tf.exp(logp_all), axis=-1))    #[B, A] => [B,]
                    alpha_loss = -tf.reduce_mean(self.alpha * corr)

            critic_grads = tape.gradient(critic_loss, self.q_net.trainable_variables + self.v_net.trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.q_net.trainable_variables + self.v_net.trainable_variables)
            )
            actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.trainable_variables)
            )
            if self.auto_adaption:
                alpha_grad = tape.gradient(alpha_loss, self.log_alpha)
                self.optimizer_alpha.apply_gradients(
                    [(alpha_grad, self.log_alpha)]
                )
            self.global_step.assign_add(1)
            summaries = dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/q1_loss', q1_loss],
                ['LOSS/q2_loss', q2_loss],
                ['LOSS/v_loss', v_loss_stop],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/log_alpha', self.log_alpha],
                ['Statistics/alpha', self.alpha],
                ['Statistics/entropy', tf.reduce_mean(entropy)],
                ['Statistics/v_mean', tf.reduce_mean(v)]
            ])
            if self.auto_adaption:
                summaries.update({
                    'LOSS/alpha_loss': alpha_loss
                })
            return (td_error1 + td_error2) / 2, summaries
