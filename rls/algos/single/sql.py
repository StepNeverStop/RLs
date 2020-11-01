#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rls.nn import critic_q_all as NetWork
from rls.algos.base.off_policy import make_off_policy_class
from rls.utils.tf2_utils import update_target_net_weights


class SQL(make_off_policy_class(mode='share')):
    '''
        Soft Q-Learning.
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

        def _q_net(): return NetWork(self.feat_dim, self.a_dim, network_settings)

        self.q_net = _q_net()
        self.q_target_net = _q_net()
        self.critic_tv = self.q_net.trainable_variables + self.other_tv
        self.lr = self.init_lr(lr)
        self.optimizer = self.init_optimizer(self.lr)

        update_target_net_weights(
            self.q_target_net.weights,
            self.q_net.weights
        )

        self._worker_params_dict.update(model=self.q_net)
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
            q_values = self.q_net(feat)
            # NOTE: check whether this is correct or not
            logits = tf.math.exp((q_values - self.get_v(q_values)) / self.alpha)    # > 0
            cate_dist = tfp.distributions.Categorical(logits=tf.nn.log_softmax(logits))
            pi = cate_dist.sample()
        return pi, cell_state

    @tf.function
    def get_v(self, q):
        with tf.device(self.device):
            v = self.alpha * tf.math.log(tf.reduce_mean(tf.math.exp(q / self.alpha), axis=1, keepdims=True))
        return v

    def _target_params_update(self): 
        update_target_net_weights(
            self.q_target_net.weights,
            self.q_net.weights,
            self.ployak)
        
    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([['LEARNING_RATE/lr', self.lr(self.train_step)]])
            })

    @tf.function(experimental_relax_shapes=True)
    def _train(self, memories, isw, cell_state):
        ss, vvss, a, r, done = memories
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                q = self.q_net(feat)
                q_next = self.q_target_net(feat_)
                v_next = self.get_v(q_next)
                q_eval = tf.reduce_sum(tf.multiply(q, a), axis=1, keepdims=True)
                q_target = tf.stop_gradient(r + self.gamma * (1 - done) * v_next)
                td_error = q_eval - q_target
                q_loss = tf.reduce_mean(tf.square(td_error) * isw)
            grads = tape.gradient(q_loss, self.critic_tv)
            self.optimizer.apply_gradients(
                zip(grads, self.critic_tv)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/loss', q_loss],
                ['Statistics/q_max', tf.reduce_max(q_eval)],
                ['Statistics/q_min', tf.reduce_min(q_eval)],
                ['Statistics/q_mean', tf.reduce_mean(q_eval)]
            ])
