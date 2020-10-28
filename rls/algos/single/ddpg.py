#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rls.nn import actor_dpg as ActorCts
from rls.nn import actor_discrete as ActorDcs
from rls.nn import critic_q_one as Critic
from rls.nn.noise import (OrnsteinUhlenbeckActionNoise,
                          NormalActionNoise)
from rls.algos.base.off_policy import make_off_policy_class
from rls.utils.tf2_utils import update_target_net_weights


class DDPG(make_off_policy_class(mode='share')):
    '''
    Deep Deterministic Policy Gradient, https://arxiv.org/abs/1509.02971
    '''

    def __init__(self,
                 envspec,

                 ployak=0.995,
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
        self.discrete_tau = discrete_tau

        if self.is_continuous:
            def _actor_net(): return ActorCts(self.feat_dim, self.a_dim, network_settings['actor_continuous'])
            # self.action_noise = NormalActionNoise(mu=np.zeros(self.a_dim), sigma=1 * np.ones(self.a_dim))
            self.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_dim), sigma=0.2 * np.ones(self.a_dim))
        else:
            def _actor_net(): return ActorDcs(self.feat_dim, self.a_dim, network_settings['actor_discrete'])
            self.gumbel_dist = tfp.distributions.Gumbel(0, 1)

        self.actor_net = _actor_net()
        self.actor_target_net = _actor_net()
        self.actor_tv = self.actor_net.trainable_variables

        def _q_net(): return Critic(self.feat_dim, self.a_dim, network_settings['q'])
        self.q_net = _q_net()
        self.q_target_net = _q_net()
        self.critic_tv = self.q_net.trainable_variables + self.other_tv
        update_target_net_weights(
            self.actor_target_net.weights + self.q_target_net.weights,
            self.actor_net.weights + self.q_net.weights
        )
        self.actor_lr, self.critic_lr = map(self.init_lr, [actor_lr, critic_lr])
        self.optimizer_actor, self.optimizer_critic = map(self.init_optimizer, [self.actor_lr, self.critic_lr])

        self._worker_params_dict.update(actor=self.actor_net)
        self._residual_params_dict.update(
            critic=self.q_net,
            optimizer_actor=self.optimizer_actor,
            optimizer_critic=self.optimizer_critic)
        self._model_post_process()

    def choose_action(self, s, visual_s, evaluation=False):
        mu, pi, self.cell_state = self._get_action(s, visual_s, self.cell_state)
        a = mu.numpy() if evaluation else pi.numpy()
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state):
        with tf.device(self.device):
            feat, cell_state = self.get_feature(s, visual_s, cell_state=cell_state, record_cs=True)
            if self.is_continuous:
                mu = self.actor_net(feat)
                pi = tf.clip_by_value(mu + self.action_noise(), -1, 1)
            else:
                logits = self.actor_net(feat)
                mu = tf.argmax(logits, axis=1)
                cate_dist = tfp.distributions.Categorical(logits=tf.nn.log_softmax(logits))
                pi = cate_dist.sample()
            return mu, pi, cell_state

    def _target_params_update(self):
        update_target_net_weights(
            self.actor_target_net.weights + self.q_target_net.weights,
            self.actor_net.weights + self.q_net.weights,
            self.ployak)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([
                    ['LEARNING_RATE/actor_lr', self.actor_lr(self.train_step)],
                    ['LEARNING_RATE/critic_lr', self.critic_lr(self.train_step)]
                ])
            })

    @tf.function(experimental_relax_shapes=True)
    def _train(self, memories, isw, crsty_loss, cell_state):
        ss, vvss, a, r, done = memories
        batch_size = tf.shape(a)[0]
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                if self.is_continuous:
                    target_mu = self.actor_target_net(feat_)
                    action_target = tf.clip_by_value(target_mu + self.action_noise(), -1, 1)
                else:
                    target_logits = self.actor_target_net(feat_)
                    logp_all = tf.nn.log_softmax(target_logits)
                    gumbel_noise = tf.cast(self.gumbel_dist.sample([batch_size, self.a_dim]), dtype=tf.float32)
                    _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_dim)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    action_target = _pi_diff + _pi
                q = self.q_net(feat, a)
                q_target = self.q_target_net(feat_, action_target)
                dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                td_error = q - dc_r
                q_loss = 0.5 * tf.reduce_mean(tf.square(td_error) * isw) + crsty_loss
            q_grads = tape.gradient(q_loss, self.critic_tv)
            self.optimizer_critic.apply_gradients(
                zip(q_grads, self.critic_tv)
            )
            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu = self.actor_net(feat)
                else:
                    logits = self.actor_net(feat)
                    _pi = tf.nn.softmax(logits)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(logits, axis=-1), self.a_dim, dtype=tf.float32)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    mu = _pi_diff + _pi
                q_actor = self.q_net(feat, mu)
                actor_loss = -tf.reduce_mean(q_actor)
            actor_grads = tape.gradient(actor_loss, self.actor_tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_tv)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', q_loss],
                ['Statistics/q_min', tf.reduce_min(q)],
                ['Statistics/q_mean', tf.reduce_mean(q)],
                ['Statistics/q_max', tf.reduce_max(q)]
            ])

    @tf.function(experimental_relax_shapes=True)
    def train_persistent(self, memories, isw, crsty_loss, cell_state):
        ss, vvss, a, r, done = memories
        batch_size = tf.shape(a)[0]
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                if self.is_continuous:
                    target_mu = self.actor_target_net(feat_)
                    action_target = tf.clip_by_value(target_mu + self.action_noise(), -1, 1)
                    mu = self.actor_net(feat)
                else:
                    target_logits = self.actor_target_net(feat_)
                    logp_all = tf.nn.log_softmax(target_logits)
                    gumbel_noise = tf.cast(self.gumbel_dist.sample([batch_size, self.a_dim]), dtype=tf.float32)
                    _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_dim)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    action_target = _pi_diff + _pi
                    logits = self.actor_net(feat)
                    _pi = tf.nn.softmax(logits)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(logits, axis=-1), self.a_dim, dtype=tf.float32)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    mu = _pi_diff + _pi
                q = self.q_net(feat, a)
                q_target = self.q_target_net(feat_, action_target)
                dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                td_error = q - dc_r
                q_loss = 0.5 * tf.reduce_mean(tf.square(td_error) * isw) + crsty_loss

                q_actor = self.q_net(feat, mu)
                actor_loss = -tf.reduce_mean(q_actor)
            q_grads = tape.gradient(q_loss, self.critic_tv)
            self.optimizer_critic.apply_gradients(
                zip(q_grads, self.critic_tv)
            )
            actor_grads = tape.gradient(actor_loss, self.actor_tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_tv)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', q_loss],
                ['Statistics/q_min', tf.reduce_min(q)],
                ['Statistics/q_mean', tf.reduce_mean(q)],
                ['Statistics/q_max', tf.reduce_max(q)]
            ])
