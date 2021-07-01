#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rls.algos.base.off_policy import Off_Policy
from rls.utils.tf2_utils import update_target_net_weights
from rls.utils.build_networks import ValueNetwork
from rls.utils.specs import OutputNetworkType


class SQL(Off_Policy):
    '''
        Soft Q-Learning. ref: https://github.com/Bigpig4396/PyTorch-Soft-Q-Learning/blob/master/SoftQ.py
        NOTE: not the original of the paper, NO SVGD.
        Reinforcement Learning with Deep Energy-Based Policies: https://arxiv.org/abs/1702.08165
    '''

    def __init__(self,
                 envspec,

                 lr=5.0e-4,
                 alpha=2,
                 ployak=0.995,
                 network_settings=[32, 32],
                 **kwargs):
        assert not envspec.is_continuous, 'sql only support discrete action space'
        super().__init__(envspec=envspec, **kwargs)
        self.alpha = alpha
        self.ployak = ployak

        def _create_net(name, representation_net=None): return ValueNetwork(
            name=name,
            representation_net=representation_net,
            value_net_type=OutputNetworkType.CRITIC_QVALUE_ALL,
            value_net_kwargs=dict(output_shape=self.a_dim, network_settings=network_settings)
        )

        self.q_net = _create_net('q_net', self._representation_net)
        self._representation_target_net = self._create_representation_net('_representation_target_net')
        self.q_target_net = _create_net('q_target_net', self._representation_target_net)
        self.lr = self.init_lr(lr)
        self.optimizer = self.init_optimizer(self.lr)

        update_target_net_weights(self.q_target_net.weights, self.q_net.weights)

        self._worker_params_dict.update(self.q_net._policy_models)

        self._all_params_dict.update(self.q_net._all_models)
        self._all_params_dict.update(optimizer=self.optimizer)
        self._model_post_process()
        self.initialize_data_buffer()

    def choose_action(self, obs, evaluation=False):
        a, self.cell_state = self._get_action(obs, self.cell_state)
        a = a.numpy()
        return a

    @tf.function
    def _get_action(self, obs, cell_state):
        with tf.device(self.device):
            q_values, cell_state = self.q_net(obs, cell_state=cell_state)
            logits = tf.math.exp((q_values - self.get_v(q_values)) / self.alpha)    # > 0
            logits /= tf.reduce_sum(logits)
            cate_dist = tfp.distributions.Categorical(logits=logits)
            pi = cate_dist.sample()
        return pi, cell_state

    @tf.function
    def get_v(self, q):
        with tf.device(self.device):
            v = self.alpha * tf.math.log(tf.reduce_mean(tf.math.exp(q / self.alpha), axis=1, keepdims=True))
        return v

    def _target_params_update(self):
        update_target_net_weights(self.q_target_net.weights, self.q_net.weights, self.ployak)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([['LEARNING_RATE/lr', self.lr(self.train_step)]])
            })

    @tf.function
    def _train(self, BATCH, isw, cell_state):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                q, _ = self.q_net(BATCH.obs, cell_state=cell_state)
                q_next, _ = self.q_target_net(BATCH.obs_, cell_state=cell_state)
                v_next = self.get_v(q_next)
                q_eval = tf.reduce_sum(tf.multiply(q, BATCH.action), axis=1, keepdims=True)
                q_target = tf.stop_gradient(BATCH.reward + self.gamma * (1 - BATCH.done) * v_next)
                td_error = q_target - q_eval
                q_loss = tf.reduce_mean(tf.square(td_error) * isw)
            grads = tape.gradient(q_loss, self.q_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.q_net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/loss', q_loss],
                ['Statistics/q_max', tf.reduce_max(q_eval)],
                ['Statistics/q_min', tf.reduce_min(q_eval)],
                ['Statistics/q_mean', tf.reduce_mean(q_eval)]
            ])
