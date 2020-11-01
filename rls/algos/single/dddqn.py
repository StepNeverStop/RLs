#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf

from rls.nn import critic_dueling as NetWork
from rls.algos.base.off_policy import make_off_policy_class
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.tf2_utils import update_target_net_weights


class DDDQN(make_off_policy_class(mode='share')):
    '''
    Dueling Double DQN, https://arxiv.org/abs/1511.06581
    '''

    def __init__(self,
                 envspec,

                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=2,
                 network_settings={
                     'share': [128],
                     'v': [128],
                     'adv': [128]
                 },
                 **kwargs):
        assert not envspec.is_continuous, 'dueling double dqn only support discrete action space'
        super().__init__(envspec=envspec, **kwargs)
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval

        def _net(): return NetWork(self.feat_dim, self.a_dim, network_settings)

        self.dueling_net = _net()
        self.dueling_target_net = _net()
        self.critic_tv = self.dueling_net.trainable_variables + self.other_tv
        update_target_net_weights(self.dueling_target_net.weights, self.dueling_net.weights)
        self.lr = self.init_lr(lr)
        self.optimizer = self.init_optimizer(self.lr)

        self._worker_params_dict.update(model=self.dueling_net)
        self._residual_params_dict.update(optimizer=self.optimizer)
        self._model_post_process()

    def choose_action(self, s, visual_s, evaluation=False):
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
            q = self.dueling_net(feat)
        return tf.argmax(q, axis=-1), cell_state

    def _target_params_update(self):
        if self.global_step % self.assign_interval == 0:
            update_target_net_weights(self.dueling_target_net.weights, self.dueling_net.weights)

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
                q = self.dueling_net(feat)
                q_eval = tf.reduce_sum(tf.multiply(q, a), axis=1, keepdims=True)
                next_q = self.dueling_net(feat_)
                next_max_action = tf.argmax(next_q, axis=1, name='next_action_int')
                next_max_action_one_hot = tf.one_hot(tf.squeeze(next_max_action), self.a_dim, 1., 0., dtype=tf.float32)
                next_max_action_one_hot = tf.cast(next_max_action_one_hot, tf.float32)
                q_target = self.dueling_target_net(feat_)

                q_target_next_max = tf.reduce_sum(
                    tf.multiply(q_target, next_max_action_one_hot),
                    axis=1, keepdims=True)
                q_target = tf.stop_gradient(r + self.gamma * (1 - done) * q_target_next_max)
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
