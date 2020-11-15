#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf

from typing import (Union,
                    List,
                    NoReturn)

from rls.algos.single.dqn import DQN


class DDQN(DQN):
    '''
    Double DQN, https://arxiv.org/abs/1509.06461
    Double DQN + LSTM, https://arxiv.org/abs/1908.06040
    '''

    def __init__(self,
                 envspec,

                 lr: float = 5.0e-4,
                 eps_init: float = 1,
                 eps_mid: float = 0.2,
                 eps_final: float = 0.01,
                 init2mid_annealing_step: int = 1000,
                 assign_interval: int = 2,
                 network_settings: List = [32, 32],
                 **kwargs):
        assert not envspec.is_continuous, 'double dqn only support discrete action space'
        super().__init__(
            envspec=envspec,
            lr=lr,
            eps_init=eps_init,
            eps_mid=eps_mid,
            eps_final=eps_final,
            init2mid_annealing_step=init2mid_annealing_step,
            assign_interval=assign_interval,
            network_settings=network_settings,
            **kwargs)

    def learn(self, **kwargs) -> NoReturn:
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([['LEARNING_RATE/lr', self.lr(self.train_step)]]),
                'train_data_list': ['ss', 'vvss', 'a', 'r', 'done', 's_', 'visual_s_']
            })

    @tf.function(experimental_relax_shapes=True)
    def _train(self, memories, isw, cell_state):
        ss, vvss, a, r, done, s_, visual_s_ = memories
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                (feat, feat_), _ = self._representation_net(ss, vvss, cell_state=cell_state, need_split=True)
                q = self.q_net.value_net(feat)
                q_next = self.q_net.value_net(feat_)
                q_target_next, _ = self.q_target_net(s_, visual_s_, cell_state=cell_state)
                next_max_action = tf.argmax(q_next, axis=1)
                next_max_action_one_hot = tf.one_hot(tf.squeeze(next_max_action), self.a_dim, 1., 0., dtype=tf.float32)
                next_max_action_one_hot = tf.cast(next_max_action_one_hot, tf.float32)
                q_eval = tf.reduce_sum(tf.multiply(q, a), axis=1, keepdims=True)
                q_target_next_max = tf.reduce_sum(tf.multiply(q_target_next, next_max_action_one_hot), axis=1, keepdims=True)
                q_target = tf.stop_gradient(r + self.gamma * (1 - done) * q_target_next_max)
                td_error = q_eval - q_target
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
