#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rls.nn import actor_mu as ActorCts
from rls.nn import actor_discrete as ActorDcs
from rls.utils.tf2_utils import \
    get_TensorSpecs, \
    gaussian_clip_rsample, \
    gaussian_likelihood_sum, \
    gaussian_entropy
from rls.algos.base.on_policy import make_on_policy_class


class PG(make_on_policy_class(mode='share')):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim,
                 is_continuous,

                 lr=5.0e-4,
                 epoch=5,
                 hidden_units={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32]
                 },
                 **kwargs):
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim=a_dim,
            is_continuous=is_continuous,
            **kwargs)
        self.epoch = epoch
        # self.TensorSpecs = get_TensorSpecs([self.s_dim], self.visual_dim, [self.a_dim], [1])
        if self.is_continuous:
            self.net = ActorCts(self.feat_dim, self.a_dim, hidden_units['actor_continuous'])
            self.log_std = tf.Variable(initial_value=-0.5 * np.ones(self.a_dim, dtype=np.float32), trainable=True)
            self.net_tv = self.net.trainable_variables + [self.log_std] + self.other_tv
        else:
            self.net = ActorDcs(self.feat_dim, self.a_dim, hidden_units['actor_discrete'])
            self.net_tv = self.net.trainable_variables + self.other_tv
        self.lr = self.init_lr(lr)
        self.optimizer = self.init_optimizer(self.lr)

        self.initialize_data_buffer()

        self._worker_params_dict.update(model=self.net)
        self._residual_params_dict.update(optimizer=self.optimizer)
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
                mu = self.net(feat)
                sample_op, _ = gaussian_clip_rsample(mu, self.log_std)
            else:
                logits = self.net(feat)
                norm_dist = tfp.distributions.Categorical(logits)
                sample_op = norm_dist.sample()
        return sample_op, cell_state

    def calculate_statistics(self):
        self.data.cal_dc_r(self.gamma, 0., normalize=True)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        def _train(data, crsty_loss, cell_state):
            for _ in range(self.epoch):
                loss, entropy = self.train(
                    data,
                    crsty_loss,
                    cell_state
                )
            summaries = dict([
                ['LOSS/loss', loss],
                ['Statistics/entropy', entropy]
            ])
            return summaries

        self._learn(function_dict={
            'calculate_statistics': self.calculate_statistics,
            'train_function': _train,
            'train_data_list': ['s', 'visual_s', 'a', 'discounted_reward'],
            'summary_dict': dict([['LEARNING_RATE/lr', self.lr(self.train_step)]])
        })

    @tf.function(experimental_relax_shapes=True)
    def train(self, memories, crsty_loss, cell_state):
        s, visual_s, a, dc_r = memories
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat = self.get_feature(s, visual_s, cell_state=cell_state)
                if self.is_continuous:
                    mu = self.net(feat)
                    log_act_prob = gaussian_likelihood_sum(a, mu, self.log_std)
                    entropy = gaussian_entropy(self.log_std)
                else:
                    logits = self.net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    log_act_prob = tf.reduce_sum(tf.multiply(logp_all, a), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                loss = -tf.reduce_mean(log_act_prob * dc_r) + crsty_loss
            loss_grads = tape.gradient(loss, self.net_tv)
            self.optimizer.apply_gradients(
                zip(loss_grads, self.net_tv)
            )
            self.global_step.assign_add(1)
            return loss, entropy
