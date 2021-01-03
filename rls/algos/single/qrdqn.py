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


class QRDQN(Off_Policy):
    '''
    Quantile Regression DQN
    Distributional Reinforcement Learning with Quantile Regression, https://arxiv.org/abs/1710.10044
    No double, no dueling, no noisy net.
    '''

    def __init__(self,
                 envspec,

                 nums=20,
                 huber_delta=1.,
                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=1000,
                 network_settings=[128, 128],
                 **kwargs):
        assert not envspec.is_continuous, 'qrdqn only support discrete action space'
        assert nums > 0
        super().__init__(envspec=envspec, **kwargs)
        self.nums = nums
        self.huber_delta = huber_delta
        self.quantiles = tf.reshape(tf.constant((2 * np.arange(self.nums) + 1) / (2.0 * self.nums), dtype=tf.float32), [-1, self.nums])  # [1, N]
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval

        def _create_net(name, representation_net=None): return ValueNetwork(
            name=name,
            representation_net=representation_net,
            value_net_type=OutputNetworkType.QRDQN_DISTRIBUTIONAL,
            value_net_kwargs=dict(action_dim=self.a_dim, nums=self.nums, network_settings=network_settings)
        )

        self.q_dist_net = _create_net('q_dist_net', self._representation_net)
        self._representation_target_net = self._create_representation_net('_representation_target_net')
        self.q_target_dist_net = _create_net('q_target_dist_net', self._representation_target_net)
        update_target_net_weights(self.q_target_dist_net.weights, self.q_dist_net.weights)
        self.lr = self.init_lr(lr)
        self.optimizer = self.init_optimizer(self.lr)

        self._worker_params_dict.update(self.q_dist_net._policy_models)

        self._all_params_dict.update(self.q_dist_net._all_models)
        self._all_params_dict.update(optimizer=self.optimizer)
        self._model_post_process()

    def choose_action(self, obs, evaluation=False):
        if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
            a = np.random.randint(0, self.a_dim, self.n_agents)
        else:
            a, self.cell_state = self._get_action(obs, self.cell_state)
            a = a.numpy()
        return a

    @tf.function
    def _get_action(self, obs, cell_state):
        with tf.device(self.device):
            q_values, cell_state = self.q_dist_net(obs, cell_state=cell_state) 
            q = tf.reduce_mean(q_values, axis=-1)  # [B, A, N] => [B, A]
        return tf.argmax(q, axis=-1), cell_state  # [B, 1]

    def _target_params_update(self):
        if self.global_step % self.assign_interval == 0:
            update_target_net_weights(self.q_target_dist_net.weights, self.q_dist_net.weights)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([['LEARNING_RATE/lr', self.lr(self.train_step)]])
            })

    @tf.function(experimental_relax_shapes=True)
    def _train(self, memories, isw, cell_state):
        batch_size = tf.shape(memories.action)[0]
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                indexes = tf.reshape(tf.range(batch_size), [-1, 1])  # [B, 1]
                q_dist, _ = self.q_dist_net(memories.obs, cell_state=cell_state)  # [B, A, N]
                q_dist = tf.transpose(tf.reduce_sum(tf.transpose(q_dist, [2, 0, 1]) * memories.action, axis=-1), [1, 0])  # [B, N]

                target_q_dist, _ = self.q_target_dist_net(memories.obs_, cell_state=cell_state)  # [B, A, N]
                target_q = tf.reduce_mean(target_q_dist, axis=-1)  # [B, A, N] => [B, A]
                a_ = tf.reshape(tf.cast(tf.argmax(target_q, axis=-1), dtype=tf.int32), [-1, 1])  # [B, 1]
                target_q_dist = tf.gather_nd(target_q_dist, tf.concat([indexes, a_], axis=-1))   # [B, N]
                target = tf.tile(memories.reward, tf.constant([1, self.nums])) \
                    + self.gamma * tf.multiply(target_q_dist,   # [1, N]
                                               (1.0 - tf.tile(memories.done, tf.constant([1, self.nums]))))  # [B, N], [B, N]* [B, N] = [B, N]

                q_eval = tf.reduce_mean(q_dist, axis=-1)    # [B, 1]
                q_target = tf.reduce_mean(target, axis=-1)  # [B, 1]
                td_error = q_target - q_eval     # [B, 1], used for PER

                quantile_error = tf.expand_dims(target, axis=1) - tf.expand_dims(q_dist, axis=-1)   # [B, 1, N] - [B, N, 1] => [B, N, N]
                huber = huber_loss(quantile_error, delta=self.huber_delta)  # [B, N, N]
                huber_abs = tf.abs(self.quantiles - tf.where(quantile_error < 0, 1., 0.))   # [1, N] - [B, N, N] => [B, N, N]
                loss = tf.reduce_mean(huber_abs * huber, axis=-1)  # [B, N, N] => [B, N]
                loss = tf.reduce_sum(loss, axis=-1)  # [B, N] => [B, ]
                loss = tf.reduce_mean(loss * isw)  # [B, ] => 1
            grads = tape.gradient(loss, self.q_dist_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.q_dist_net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/loss', loss],
                ['Statistics/q_max', tf.reduce_max(q_eval)],
                ['Statistics/q_min', tf.reduce_min(q_eval)],
                ['Statistics/q_mean', tf.reduce_mean(q_eval)]
            ])
