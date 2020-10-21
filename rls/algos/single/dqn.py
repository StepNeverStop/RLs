#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf

from typing import (Union,
                    List,
                    NoReturn)

from rls.nn import critic_q_all as NetWork
from rls.algos.base.off_policy import make_off_policy_class
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.tf2_utils import update_target_net_weights


class DQN(make_off_policy_class(mode='share')):
    '''
    Deep Q-learning Network, DQN, [2013](https://arxiv.org/pdf/1312.5602.pdf), [2015](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    DQN + LSTM, https://arxiv.org/abs/1507.06527
    '''

    def __init__(self,
                 envspec,

                 lr: float = 5.0e-4,
                 eps_init: float = 1,
                 eps_mid: float = 0.2,
                 eps_final: float = 0.01,
                 init2mid_annealing_step: int = 1000,
                 assign_interval: int = 1000,
                 network_settings: List[int] = [32, 32],
                 **kwargs):
        assert not envspec.is_continuous, 'dqn only support discrete action space'
        super().__init__(envspec=envspec, **kwargs)
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval

        def _q_net(): return NetWork(self.feat_dim, self.a_dim, network_settings)

        self.q_net = _q_net()
        self.q_target_net = _q_net()
        self.critic_tv = self.q_net.trainable_variables + self.other_tv
        update_target_net_weights(self.q_target_net.weights, self.q_net.weights)
        self.lr = self.init_lr(lr)
        self.optimizer = self.init_optimizer(self.lr)

        self._worker_params_dict.update(model=self.q_net)
        self._residual_params_dict.update(optimizer=self.optimizer)
        self._model_post_process()

    def choose_action(self,
                      s: np.ndarray,
                      visual_s: np.ndarray,
                      evaluation: bool = False) -> np.ndarray:
        if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
            a = np.random.randint(0, self.a_dim, self.n_agents)
        else:
            a, self.cell_state = self._get_action(s, visual_s, self.cell_state)
            a = a.numpy()
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state):
        with tf.device(self.device):
            feat, cell_state = self.get_feature(s, visual_s, cell_state=cell_state, record_cs=True)
            q_values = self.q_net(feat)
        return tf.argmax(q_values, axis=1), cell_state

    def _target_params_update(self):
        if self.global_step % self.assign_interval == 0:
            update_target_net_weights(self.q_target_net.weights, self.q_net.weights)

    def learn(self, **kwargs) -> NoReturn:
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([['LEARNING_RATE/lr', self.lr(self.train_step)]])
            })

    @tf.function(experimental_relax_shapes=True)
    def _train(self, memories, isw, crsty_loss, cell_state):
        ss, vvss, a, r, done = memories
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                q = self.q_net(feat)
                q_next = self.q_target_net(feat_)
                q_eval = tf.reduce_sum(tf.multiply(q, a), axis=1, keepdims=True)
                q_target = tf.stop_gradient(r + self.gamma * (1 - done) * tf.reduce_max(q_next, axis=1, keepdims=True))
                td_error = q_eval - q_target
                q_loss = tf.reduce_mean(tf.square(td_error) * isw) + crsty_loss
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

    @tf.function(experimental_relax_shapes=True)
    def _cal_td(self, memories, cell_state):
        ss, vvss, a, r, done = memories
        with tf.device(self.device):
            feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
            q = self.q_net(feat)
            q_next = self.q_target_net(feat_)
            q_eval = tf.reduce_sum(tf.multiply(q, a), axis=1, keepdims=True)
            q_target = tf.stop_gradient(r + self.gamma * (1 - done) * tf.reduce_max(q_next, axis=1, keepdims=True))
            td_error = q_eval - q_target
        return td_error

    def apex_learn(self, train_step, data, priorities):
        self.train_step = train_step
        return self._apex_learn(function_dict={
            'summary_dict': dict([['LEARNING_RATE/lr', self.lr(self.train_step)]])
        }, data=data, priorities=priorities)

    def apex_cal_td(self, data):
        return self._apex_cal_td(data=data)
