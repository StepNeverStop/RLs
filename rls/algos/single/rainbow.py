#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf

from rls.nn import RainbowDueling as NetWork
from rls.algos.base.off_policy import Off_Policy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.tf2_utils import update_target_net_weights
from rls.utils.build_networks import ValueNetwork
from rls.utils.specs import OutputNetworkType


class RAINBOW(Off_Policy):
    '''
    Rainbow DQN:    https://arxiv.org/abs/1710.02298
        1. Double
        2. Dueling
        3. PrioritizedExperienceReplay
        4. N-Step
        5. Distributional
        6. Noisy Net
    '''

    def __init__(self,
                 envspec,

                 v_min=-10,
                 v_max=10,
                 atoms=51,
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
        assert not envspec.is_continuous, 'rainbow only support discrete action space'
        super().__init__(envspec=envspec, **kwargs)
        self.v_min = v_min
        self.v_max = v_max
        self.atoms = atoms
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self.z = tf.reshape(tf.constant([self.v_min + i * self.delta_z for i in range(self.atoms)], dtype=tf.float32), [-1, self.atoms])  # [1, N]
        self.zb = tf.tile(self.z, tf.constant([self.a_dim, 1]))  # [A, N]
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval

        def _create_net(name, representation_net=None): return ValueNetwork(
            name=name,
            representation_net=representation_net,
            value_net_type=OutputNetworkType.RAINBOW_DUELING,
            value_net_kwargs=dict(action_dim=self.a_dim, atoms=self.atoms, network_settings=network_settings)
        )

        self.rainbow_net = _create_net('rainbow_net', self._representation_net)
        self._representation_target_net = self._create_representation_net('_representation_target_net')
        self.rainbow_target_net = _create_net('rainbow_target_net', self._representation_target_net)
        update_target_net_weights(self.rainbow_target_net.weights, self.rainbow_net.weights)
        self.lr = self.init_lr(lr)
        self.optimizer = self.init_optimizer(self.lr)

        self._worker_params_dict.update(self.rainbow_net._policy_models)

        self._all_params_dict.update(self.rainbow_net._all_models)
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
            q_values, cell_state = self.rainbow_net(obs, cell_state=cell_state)
            q = tf.reduce_sum(self.zb * q_values, axis=-1)  # [B, A, N] => [B, A]
        return tf.argmax(q, axis=-1), cell_state  # [B, 1]

    def _target_params_update(self):
        if self.global_step % self.assign_interval == 0:
            update_target_net_weights(self.rainbow_target_net.weights, self.rainbow_net.weights)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([['LEARNING_RATE/lr', self.lr(self.train_step)]]),
                'use_stack': True
            })

    @tf.function(experimental_relax_shapes=True)
    def _train(self, memories, isw, cell_state):
        batch_size = tf.shape(memories.action)[0]
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                (feat, feat_), _ = self._representation_net(memories.obs, cell_state=cell_state, need_split=True)
                indexes = tf.reshape(tf.range(batch_size), [-1, 1])  # [B, 1]
                q_dist = self.rainbow_net.value_net(feat)  # [B, A, N]
                q_dist = tf.transpose(tf.reduce_sum(tf.transpose(q_dist, [2, 0, 1]) * memories.action, axis=-1), [1, 0])  # [B, N]
                q_eval = tf.reduce_sum(q_dist * self.z, axis=-1)
                target_q = self.rainbow_net.value_net(feat_)
                target_q = tf.reduce_sum(self.zb * target_q, axis=-1)  # [B, A, N] => [B, A]
                a_ = tf.reshape(tf.cast(tf.argmax(target_q, axis=-1), dtype=tf.int32), [-1, 1])  # [B, 1]

                target_q_dist, _ = self.rainbow_target_net(memories.obs_, cell_state=cell_state)  # [B, A, N]
                target_q_dist = tf.gather_nd(target_q_dist, tf.concat([indexes, a_], axis=-1))   # [B, N]
                target = tf.tile(memories.reward, tf.constant([1, self.atoms])) \
                    + self.gamma * tf.multiply(self.z,   # [1, N]
                                               (1.0 - tf.tile(memories.done, tf.constant([1, self.atoms]))))  # [B, N], [1, N]* [B, N] = [B, N]
                target = tf.clip_by_value(target, self.v_min, self.v_max)  # [B, N]
                b = (target - self.v_min) / self.delta_z  # [B, N]
                u, l = tf.math.ceil(b), tf.math.floor(b)  # [B, N]
                u_id, l_id = tf.cast(u, tf.int32), tf.cast(l, tf.int32)  # [B, N]
                u_minus_b, b_minus_l = u - b, b - l  # [B, N]
                index_help = tf.tile(indexes, tf.constant([1, self.atoms]))  # [B, N]
                index_help = tf.expand_dims(index_help, -1)  # [B, N, 1]
                u_id = tf.concat([index_help, tf.expand_dims(u_id, -1)], axis=-1)    # [B, N, 2]
                l_id = tf.concat([index_help, tf.expand_dims(l_id, -1)], axis=-1)    # [B, N, 2]
                _cross_entropy = tf.stop_gradient(target_q_dist * u_minus_b) * tf.math.log(tf.gather_nd(q_dist, l_id)) \
                    + tf.stop_gradient(target_q_dist * b_minus_l) * tf.math.log(tf.gather_nd(q_dist, u_id))  # [B, N]
                cross_entropy = -tf.reduce_sum(_cross_entropy, axis=-1)  # [B,]
                loss = tf.reduce_mean(cross_entropy * isw)
                td_error = cross_entropy
            grads = tape.gradient(loss, self.rainbow_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.rainbow_net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/loss', loss],
                ['Statistics/q_max', tf.reduce_max(q_eval)],
                ['Statistics/q_min', tf.reduce_min(q_eval)],
                ['Statistics/q_mean', tf.reduce_mean(q_eval)]
            ])
