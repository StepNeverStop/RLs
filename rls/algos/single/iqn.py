#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf

from rls.algos.base.off_policy import Off_Policy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.tf2_utils import (huber_loss,
                                 update_target_net_weights)
from rls.utils.build_networks import ValueNetwork
from rls.utils.specs import OutputNetworkType


class IQN(Off_Policy):
    '''
    Implicit Quantile Networks, https://arxiv.org/abs/1806.06923
    Double DQN
    '''

    def __init__(self,
                 envspec,

                 online_quantiles=8,
                 target_quantiles=8,
                 select_quantiles=32,
                 quantiles_idx=64,
                 huber_delta=1.,
                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=2,
                 network_settings={
                     'q_net': [128, 64],
                     'quantile': [128, 64],
                     'tile': [64]
                 },
                 **kwargs):
        assert not envspec.is_continuous, 'iqn only support discrete action space'
        super().__init__(envspec=envspec, **kwargs)
        self.pi = tf.constant(np.pi)
        self.online_quantiles = online_quantiles
        self.target_quantiles = target_quantiles
        self.select_quantiles = select_quantiles
        self.quantiles_idx = quantiles_idx
        self.huber_delta = huber_delta
        self.assign_interval = assign_interval
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)

        def _create_net(name, representation_net=None): return ValueNetwork(
            name=name,
            representation_net=representation_net,
            value_net_type=OutputNetworkType.IQN_NET,
            value_net_kwargs=dict(action_dim=self.a_dim, quantiles_idx=self.quantiles_idx, network_settings=network_settings)
        )

        self.q_net = _create_net('q_net', self._representation_net)
        self._representation_target_net = self._create_representation_net('_representation_target_net')
        self.q_target_net = _create_net('q_target_net', self._representation_target_net)
        update_target_net_weights(self.q_target_net.weights, self.q_net.weights)
        self.lr = self.init_lr(lr)
        self.optimizer = self.init_optimizer(self.lr)

        self._worker_params_dict.update(self.q_net._policy_models)

        self._all_params_dict.update(self.q_net._all_models)
        self._all_params_dict.update(optimizer=self.optimizer)
        self._model_post_process()
        self.initialize_data_buffer()

    def choose_action(self, obs, evaluation=False):
        if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
            a = np.random.randint(0, self.a_dim, self.n_copys)
        else:
            a, self.cell_state = self._get_action(obs, self.cell_state)
            a = a.numpy()
        return a

    @tf.function
    def _get_action(self, obs, cell_state):
        batch_size = tf.shape(s)[0]
        with tf.device(self.device):
            _, select_quantiles_tiled = self._generate_quantiles(   # [N*B, 64]
                batch_size=batch_size,
                quantiles_num=self.select_quantiles,
                quantiles_idx=self.quantiles_idx
            )
            # [B, A]
            (_, q_values), cell_state = self.q_net(obs, select_quantiles_tiled, quantiles_num=self.select_quantiles, cell_state=cell_state)
        return tf.argmax(q_values, axis=-1), cell_state  # [B,]

    @tf.function
    def _generate_quantiles(self, batch_size, quantiles_num, quantiles_idx):
        with tf.device(self.device):
            _quantiles = tf.random.uniform([batch_size * quantiles_num, 1], minval=0, maxval=1)  # [N*B, 1]
            _quantiles_tiled = tf.tile(_quantiles, [1, quantiles_idx])  # [N*B, 1] => [N*B, 64]
            _quantiles_tiled = tf.cast(tf.range(quantiles_idx), tf.float32) * self.pi * _quantiles_tiled  # pi * i * tau [N*B, 64] * [64, ] => [N*B, 64]
            _quantiles_tiled = tf.cos(_quantiles_tiled)   # [N*B, 64]
            _quantiles = tf.reshape(_quantiles, [batch_size, quantiles_num, 1])    # [N*B, 1] => [B, N, 1]
            return _quantiles, _quantiles_tiled

    def _target_params_update(self):
        if self.global_step % self.assign_interval == 0:
            update_target_net_weights(self.q_target_net.weights, self.q_net.weights)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([['LEARNING_RATE/lr', self.lr(self.train_step)]]),
                'use_stack': True
            })

    @tf.function
    def _train(self, BATCH, isw, cell_state):
        batch_size = tf.shape(BATCH.action)[0]
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                (feat, feat_), _ = self._representation_net(BATCH.obs, cell_state=cell_state, need_split=True)
                quantiles, quantiles_tiled = self._generate_quantiles(   # [B, N, 1], [N*B, 64]
                    batch_size=batch_size,
                    quantiles_num=self.online_quantiles,
                    quantiles_idx=self.quantiles_idx
                )
                quantiles_value, q = self.q_net.value_net(feat, quantiles_tiled, quantiles_num=self.online_quantiles)    # [N, B, A], [B, A]
                _a = tf.reshape(tf.tile(BATCH.action, [self.online_quantiles, 1]), [self.online_quantiles, -1, self.a_dim])  # [B, A] => [N*B, A] => [N, B, A]
                quantiles_value = tf.reduce_sum(quantiles_value * _a, axis=-1, keepdims=True)   # [N, B, A] => [N, B, 1]
                q_eval = tf.reduce_sum(q * BATCH.action, axis=-1, keepdims=True)  # [B, A] => [B, 1]

                _, select_quantiles_tiled = self._generate_quantiles(   # [N*B, 64]
                    batch_size=batch_size,
                    quantiles_num=self.select_quantiles,
                    quantiles_idx=self.quantiles_idx
                )
                _, q_values = self.q_net.value_net(feat_, select_quantiles_tiled, quantiles_num=self.select_quantiles)  # [B, A]
                next_max_action = tf.argmax(q_values, axis=-1)   # [B,]
                next_max_action = tf.one_hot(tf.squeeze(next_max_action), self.a_dim, 1., 0., dtype=tf.float32)  # [B, A]
                _next_max_action = tf.reshape(tf.tile(next_max_action, [self.target_quantiles, 1]), [self.target_quantiles, -1, self.a_dim])  # [B, A] => [N'*B, A] => [N', B, A]
                _, target_quantiles_tiled = self._generate_quantiles(   # [N'*B, 64]
                    batch_size=batch_size,
                    quantiles_num=self.target_quantiles,
                    quantiles_idx=self.quantiles_idx
                )

                (target_quantiles_value, target_q), _ = self.q_target_net(BATCH.obs_, target_quantiles_tiled, quantiles_num=self.target_quantiles, cell_state=cell_state)  # [N', B, A], [B, A]
                target_quantiles_value = tf.reduce_sum(target_quantiles_value * _next_max_action, axis=-1, keepdims=True)   # [N', B, A] => [N', B, 1]
                target_q = tf.reduce_sum(target_q * BATCH.action, axis=-1, keepdims=True)  # [B, A] => [B, 1]
                q_target = tf.stop_gradient(BATCH.reward + self.gamma * (1 - BATCH.done) * target_q)   # [B, 1]
                td_error = q_target - q_eval    # [B, 1]

                _r = tf.reshape(tf.tile(BATCH.reward, [self.target_quantiles, 1]), [self.target_quantiles, -1, 1])  # [B, 1] => [N'*B, 1] => [N', B, 1]
                _done = tf.reshape(tf.tile(BATCH.done, [self.target_quantiles, 1]), [self.target_quantiles, -1, 1])    # [B, 1] => [N'*B, 1] => [N', B, 1]

                quantiles_value_target = tf.stop_gradient(_r + self.gamma * (1 - _done) * target_quantiles_value)   # [N', B, 1]
                quantiles_value_target = tf.transpose(quantiles_value_target, [1, 2, 0])    # [B, 1, N']
                quantiles_value_online = tf.transpose(quantiles_value, [1, 0, 2])   # [B, N, 1]
                quantile_error = quantiles_value_online - quantiles_value_target    # [B, N, 1] - [B, 1, N'] => [B, N, N']
                huber = huber_loss(quantile_error, delta=self.huber_delta)  # [B, N, N']
                huber_abs = tf.abs(quantiles - tf.where(quantile_error < 0, tf.ones_like(quantile_error), tf.zeros_like(quantile_error)))   # [B, N, 1] - [B, N, N'] => [B, N, N']
                loss = tf.reduce_mean(huber_abs * huber, axis=-1)  # [B, N, N'] => [B, N]
                loss = tf.reduce_sum(loss, axis=-1)  # [B, N] => [B, ]
                loss = tf.reduce_mean(loss * isw)  # [B, ] => 1
            grads = tape.gradient(loss, self.q_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.q_net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/loss', loss],
                ['Statistics/q_max', tf.reduce_max(q_eval)],
                ['Statistics/q_min', tf.reduce_min(q_eval)],
                ['Statistics/q_mean', tf.reduce_mean(q_eval)]
            ])
