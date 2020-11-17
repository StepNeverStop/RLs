#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rls.algos.base.off_policy import Off_Policy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.tf2_utils import update_target_net_weights
from rls.utils.build_networks import ValueNetwork
from rls.utils.indexs import OutputNetworkType


class BootstrappedDQN(Off_Policy):
    '''
    Deep Exploration via Bootstrapped DQN, http://arxiv.org/abs/1602.04621
    '''

    def __init__(self,
                 envspec,

                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=1000,
                 head_num=4,
                 network_settings=[32, 32],
                 **kwargs):
        assert not envspec.is_continuous, 'Bootstrapped DQN only support discrete action space'
        super().__init__(envspec=envspec, **kwargs)
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval
        self.head_num = head_num
        self._probs = [1. / head_num for _ in range(head_num)]
        self.now_head = 0

        def _create_net(name, representation_net=None): return ValueNetwork(
            name=name,
            representation_net=representation_net,
            value_net_type=OutputNetworkType.CRITIC_QVALUE_BOOTSTRAP,
            value_net_kwargs=dict(output_shape=self.a_dim, head_num=self.head_num, network_settings=network_settings)
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

    def reset(self):
        super().reset()
        self.now_head = np.random.randint(self.head_num)

    def choose_action(self, s, visual_s, evaluation=False):
        if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
            a = np.random.randint(0, self.a_dim, self.n_agents)
        else:
            q, self.cell_state = self._get_action(s, visual_s, self.cell_state)
            q = q.numpy()
            a = np.argmax(q[self.now_head], axis=1)  # [H, B, A] => [B, A] => [B, ]
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state):
        with tf.device(self.device):
            q_values, cell_state = self.q_net(s, visual_s, cell_state=cell_state)  # [H, B, A]
        return q_values, cell_state

    def _target_params_update(self):
        if self.global_step % self.assign_interval == 0:
            update_target_net_weights(self.q_target_net.weights, self.q_net.weights)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([['LEARNING_RATE/lr', self.lr(self.train_step)]]),
                'train_data_list': ['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done']
            })

    @tf.function(experimental_relax_shapes=True)
    def _train(self, memories, isw, cell_state):
        s, visual_s, a, r, s_, visual_s_, done = memories
        batch_size = tf.shape(a)[0]
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                q, _ = self.q_net(s, visual_s, cell_state=cell_state)    # [H, B, A]
                q_next, _ = self.q_target_net(s_, visual_s_, cell_state=cell_state)   # [H, B, A]
                q_eval = tf.reduce_sum(tf.multiply(q, a), axis=-1, keepdims=True)    # [H, B, A] * [B, A] => [H, B, 1]
                q_target = tf.stop_gradient(r + self.gamma * (1 - done) * tf.reduce_max(q_next, axis=-1, keepdims=True))
                td_error = q_eval - q_target    # [H, B, 1]
                td_error = tf.reduce_sum(td_error, axis=-1)  # [H, B]

                mask_dist = tfp.distributions.Bernoulli(probs=self._probs)
                mask = tf.transpose(mask_dist.sample(batch_size), [1, 0])   # [H, B]
                q_loss = tf.reduce_mean(tf.square(td_error) * isw)
            grads = tape.gradient(q_loss, self.q_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.q_net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return tf.reduce_mean(td_error, axis=0), dict([  # [H, B] =>
                ['LOSS/loss', q_loss],
                ['Statistics/q_max', tf.reduce_max(q_eval)],
                ['Statistics/q_min', tf.reduce_min(q_eval)],
                ['Statistics/q_mean', tf.reduce_mean(q_eval)]
            ])
