#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rls.utils.tf2_utils import (squash_rsample,
                                 gaussian_entropy,
                                 update_target_net_weights)
from rls.algos.base.off_policy import Off_Policy
from rls.utils.sundry_utils import LinearAnnealing
from rls.utils.build_networks import (ValueNetwork,
                                      DoubleValueNetwork)
from rls.utils.specs import OutputNetworkType


class SAC(Off_Policy):
    """
        Soft Actor-Critic Algorithms and Applications. https://arxiv.org/abs/1812.05905
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
                 network_settings={
                     'actor_continuous': {
                         'share': [128, 128],
                         'mu': [64],
                         'log_std': [64],
                         'soft_clip': False,
                         'log_std_bound': [-20, 2]
                     },
                     'actor_discrete': [64, 32],
                     'q': [128, 128]
                 },
                 auto_adaption=True,
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 alpha_lr=5.0e-4,
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.ployak = ployak
        self.use_gumbel = use_gumbel
        self.discrete_tau = discrete_tau
        self.auto_adaption = auto_adaption
        self.annealing = annealing

        if self.auto_adaption:
            self.log_alpha = tf.Variable(initial_value=0.0, name='log_alpha', dtype=tf.float32, trainable=True)
        else:
            self.log_alpha = tf.Variable(initial_value=tf.math.log(alpha), name='log_alpha', dtype=tf.float32, trainable=False)
            if self.annealing:
                self.alpha_annealing = LinearAnnealing(alpha, last_alpha, 1e6)

        # entropy = -log(1/|A|) = log |A|
        self.target_entropy = 0.98 * (-self.a_dim if self.is_continuous else np.log(self.a_dim))

        if self.is_continuous or self.use_gumbel:
            def _create_net(name, representation_net=None): return DoubleValueNetwork(
                name=name,
                representation_net=representation_net,
                value_net_type=OutputNetworkType.CRITIC_QVALUE_ONE,
                value_net_kwargs=dict(action_dim=self.a_dim, network_settings=network_settings['q'])
            )
        else:
            def _create_net(name, representation_net=None): return DoubleValueNetwork(
                name=name,
                representation_net=representation_net,
                value_net_type=OutputNetworkType.CRITIC_QVALUE_ALL,
                value_net_kwargs=dict(output_shape=self.a_dim, network_settings=network_settings['q'])
            )

        self.critic_net = _create_net('critic_net', self._representation_net)
        self.critic_target_net = _create_net('critic_target_net', self._representation_net._copy())

        if self.is_continuous:
            self.actor_net = ValueNetwork(
                name='actor_net',
                representation_net=self._representation_net,
                train_representation_net=False,
                value_net_type=OutputNetworkType.ACTOR_CTS,
                value_net_kwargs=dict(output_shape=self.a_dim,
                                      network_settings=network_settings['actor_continuous'])
            )
        else:
            self.actor_net = ValueNetwork(
                name='actor_net',
                representation_net=self._representation_net,
                train_representation_net=False,
                value_net_type=OutputNetworkType.ACTOR_DCT,
                value_net_kwargs=dict(output_shape=self.a_dim,
                                      network_settings=network_settings['actor_discrete'])
            )
            if self.use_gumbel:
                self.gumbel_dist = tfp.distributions.Gumbel(0, 1)

        update_target_net_weights(self.critic_target_net.weights, self.critic_net.weights)
        self.actor_lr, self.critic_lr, self.alpha_lr = map(self.init_lr, [actor_lr, critic_lr, alpha_lr])
        self.optimizer_actor, self.optimizer_critic, self.optimizer_alpha = map(self.init_optimizer, [self.actor_lr, self.critic_lr, self.alpha_lr])

        self._worker_params_dict.update(self.actor_net._policy_models)

        self._all_params_dict.update(self.actor_net._all_models)
        self._all_params_dict.update(self.critic_net._all_models)
        self._all_params_dict.update(log_alpha=self.log_alpha,
                                     optimizer_actor=self.optimizer_actor,
                                     optimizer_critic=self.optimizer_critic,
                                     optimizer_alpha=self.optimizer_alpha)
        self._model_post_process()
        self.initialize_data_buffer()

    def choose_action(self, obs, evaluation=False):
        mu, pi, self.cell_state = self._get_action(obs, self.cell_state)
        a = mu.numpy() if evaluation else pi.numpy()
        return a

    @tf.function
    def _get_action(self, obs, cell_state):
        with tf.device(self.device):
            feat, cell_state = self.critic_net.get_feat(obs, cell_state=cell_state, out_cell_state=True)
            if self.is_continuous:
                mu, log_std = self.actor_net.value_net(feat)
                pi, _ = squash_rsample(mu, log_std)
                mu = tf.tanh(mu)  # squash mu
            else:
                logits = self.actor_net.value_net(feat)
                mu = tf.argmax(logits, axis=1)
                cate_dist = tfp.distributions.Categorical(logits=logits)
                pi = cate_dist.sample()
            return mu, pi, cell_state

    def _target_params_update(self):
        update_target_net_weights(self.critic_target_net.weights, self.critic_net.weights, self.ployak)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([
                    ['LEARNING_RATE/actor_lr', self.actor_lr(self.train_step)],
                    ['LEARNING_RATE/critic_lr', self.critic_lr(self.train_step)],
                    ['LEARNING_RATE/alpha_lr', self.alpha_lr(self.train_step)]
                ])
            })

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def _train(self, BATCH, isw, cell_state):
        if self.is_continuous or self.use_gumbel:
            td_error, summaries = self.train_continuous(BATCH, isw, cell_state)
        else:
            td_error, summaries = self.train_discrete(BATCH, isw, cell_state)
        if self.annealing and not self.auto_adaption:
            self.log_alpha.assign(tf.math.log(tf.cast(self.alpha_annealing(self.global_step.numpy()), tf.float32)))
        return td_error, summaries

    @tf.function
    def train_continuous(self, BATCH, isw, cell_state):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                feat = self.critic_net.get_feat(BATCH.obs, cell_state=cell_state)
                feat_ = self.critic_net.get_feat(BATCH.obs_, cell_state=cell_state)
                if self.is_continuous:
                    mu, log_std = self.actor_net.value_net(feat)
                    pi, log_pi = squash_rsample(mu, log_std)
                    entropy = gaussian_entropy(log_std)
                    target_mu, target_log_std = self.actor_net.value_net(feat_)
                    target_pi, target_log_pi = squash_rsample(target_mu, target_log_std)
                else:
                    logits = self.actor_net.value_net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    gumbel_noise = tf.cast(self.gumbel_dist.sample(BATCH.action.shape), dtype=tf.float32)
                    _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_dim)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    pi = _pi_diff + _pi
                    log_pi = tf.reduce_sum(tf.multiply(logp_all, pi), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))

                    target_logits = self.actor_net.value_net(feat_)
                    target_cate_dist = tfp.distributions.Categorical(logits=target_logits)
                    target_pi = target_cate_dist.sample()
                    target_log_pi = target_cate_dist.log_prob(target_pi)
                    target_pi = tf.one_hot(target_pi, self.a_dim, dtype=tf.float32)
                q1, q2 = self.critic_net.get_value(feat, BATCH.action)
                q_s_pi = self.critic_net.get_min(feat, pi)
                ret = self.critic_target_net(BATCH.obs_, target_pi, cell_state=cell_state)
                q1_target = ret['value']
                q1_target = ret['value2']
                q_target = tf.minimum(q1_target, q2_target)
                dc_r = tf.stop_gradient(BATCH.reward + self.gamma * (1 - BATCH.done) * (q_target - self.alpha * target_log_pi))
                td_error1 = q1 - dc_r
                td_error2 = q2 - dc_r
                q1_loss = tf.reduce_mean(tf.square(td_error1) * isw)
                q2_loss = tf.reduce_mean(tf.square(td_error2) * isw)
                critic_loss = 0.5 * q1_loss + 0.5 * q2_loss
                actor_loss = -tf.reduce_mean(q_s_pi - self.alpha * log_pi)
                if self.auto_adaption:
                    alpha_loss = -tf.reduce_mean(self.alpha * tf.stop_gradient(log_pi + self.target_entropy))
            critic_grads = tape.gradient(critic_loss, self.critic_net.trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_net.trainable_variables)
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
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/log_alpha', self.log_alpha],
                ['Statistics/alpha', self.alpha],
                ['Statistics/entropy', entropy],
                ['Statistics/q_min', tf.reduce_min(tf.minimum(q1, q2))],
                ['Statistics/q_mean', tf.reduce_mean(tf.minimum(q1, q2))],
                ['Statistics/q_max', tf.reduce_max(tf.maximum(q1, q2))]
            ])
            if self.auto_adaption:
                summaries.update({
                    'LOSS/alpha_loss': alpha_loss
                })
            return (td_error1 + td_error2) / 2, summaries

    @tf.function
    def train_discrete(self, BATCH, isw, cell_state):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                feat = self.critic_net.get_feat(BATCH.obs, cell_state=cell_state)
                feat_ = self.critic_net.get_feat(BATCH.obs_, cell_state=cell_state)
                q1_all, q2_all = self.critic_net.get_value(feat)  # [B, A]

                logits = self.actor_net.value_net(feat)
                logp_all = tf.nn.log_softmax(logits)
                entropy = -tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True)    # [B, 1]
                q_all = tf.minimum(q1_all, q2_all)  # [B, A]

                def q_function(x): return tf.reduce_sum(x * BATCH.action, axis=-1, keepdims=True)  # [B, 1]
                q1 = q_function(q1_all)
                q2 = q_function(q2_all)
                target_logits = self.actor_net.value_net(feat_)  # [B, A]
                target_log_probs = tf.nn.log_softmax(target_logits)  # [B, A]
                ret = self.critic_target_net(BATCH.obs_, cell_state=cell_state)    # [B, A]
                q1_target = ret['value']
                q1_target = ret['value2']
                def v_target_function(x): return tf.reduce_sum(tf.exp(target_log_probs) * (x - self.alpha * target_log_probs), axis=-1, keepdims=True)  # [B, 1]
                v1_target = v_target_function(q1_target)
                v2_target = v_target_function(q2_target)
                v_target = tf.minimum(v1_target, v2_target)
                dc_r = tf.stop_gradient(BATCH.reward + self.gamma * (1 - BATCH.done) * v_target)
                td_error1 = q1 - dc_r
                td_error2 = q2 - dc_r
                q1_loss = tf.reduce_mean(tf.square(td_error1) * isw)
                q2_loss = tf.reduce_mean(tf.square(td_error2) * isw)

                critic_loss = 0.5 * q1_loss + 0.5 * q2_loss
                actor_loss = -tf.reduce_mean(
                    tf.reduce_sum((q_all - self.alpha * logp_all) * tf.exp(logp_all))  # [B, A] => [B,]
                )
                # actor_loss = - tf.reduce_mean(
                #     q_all + self.alpha * entropy
                #     )
                if self.auto_adaption:
                    corr = tf.stop_gradient(self.target_entropy - entropy)
                    # corr = tf.stop_gradient(tf.reduce_sum((logp_all - self.a_dim) * tf.exp(logp_all), axis=-1))    #[B, A] => [B,]
                    # J(\alpha)=\pi_{t}\left(s_{t}\right)^{T}\left[-\alpha\left(\log \left(\pi_{t}\left(s_{t}\right)\right)+\bar{H}\right)\right]
                    # \bar{H} is negative
                    alpha_loss = -tf.reduce_mean(self.alpha * corr)

            critic_grads = tape.gradient(critic_loss, self.critic_net.trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_net.trainable_variables)
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
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/log_alpha', self.log_alpha],
                ['Statistics/alpha', self.alpha],
                ['Statistics/entropy', tf.reduce_mean(entropy)]
            ])
            if self.auto_adaption:
                summaries.update({
                    'LOSS/alpha_loss': alpha_loss
                })
            return (td_error1 + td_error2) / 2, summaries
