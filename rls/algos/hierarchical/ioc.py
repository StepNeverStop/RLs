#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rls.algos.base.off_policy import Off_Policy
from rls.utils.tf2_utils import (gaussian_clip_rsample,
                                 gaussian_likelihood_sum,
                                 gaussian_entropy,
                                 update_target_net_weights)
from rls.utils.build_networks import ValueNetwork
from rls.utils.indexs import OutputNetworkType


class IOC(Off_Policy):
    '''
    Learning Options with Interest Functions, https://www.aaai.org/ojs/index.php/AAAI/article/view/5114/4987 
    Options of Interest: Temporal Abstraction with Interest Functions, http://arxiv.org/abs/2001.00271
    '''

    def __init__(self,
                 envspec,

                 q_lr=5.0e-3,
                 intra_option_lr=5.0e-4,
                 termination_lr=5.0e-4,
                 interest_lr=5.0e-4,
                 boltzmann_temperature=1.0,
                 options_num=4,
                 ent_coff=0.01,
                 double_q=False,
                 use_baseline=True,
                 terminal_mask=True,
                 termination_regularizer=0.01,
                 assign_interval=1000,
                 network_settings={
                     'q': [32, 32],
                     'intra_option': [32, 32],
                     'termination': [32, 32],
                     'interest': [32, 32]
                 },
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.assign_interval = assign_interval
        self.options_num = options_num
        self.termination_regularizer = termination_regularizer
        self.ent_coff = ent_coff
        self.use_baseline = use_baseline
        self.terminal_mask = terminal_mask
        self.double_q = double_q
        self.boltzmann_temperature = boltzmann_temperature

        def _create_net(name, representation_net=None): return ValueNetwork(
            name=name,
            representation_net=representation_net,
            value_net_type=OutputNetworkType.CRITIC_QVALUE_ALL,
            value_net_kwargs=dict(output_shape=self.options_num, network_settings=network_settings['q'])
        )
        self.q_net = _create_net('q_net', self._representation_net)
        self._representation_target_net = self._create_representation_net('_representation_target_net')
        self.q_target_net = _create_net('q_target_net', self._representation_target_net)

        self.intra_option_net = ValueNetwork(
            name='intra_option_net',
            value_net_type=OutputNetworkType.OC_INTRA_OPTION,
            value_net_kwargs=dict(vector_dim=self._representation_net.h_dim,
                                  output_shape=self.a_dim,
                                  options_num=self.options_num,
                                  network_settings=network_settings['intra_option'])
        )
        self.termination_net = ValueNetwork(
            name='termination_net',
            value_net_type=OutputNetworkType.CRITIC_QVALUE_ALL,
            value_net_kwargs=dict(vector_dim=self._representation_net.h_dim,
                                  output_shape=self.options_num,
                                  network_settings=network_settings['termination'],
                                  out_activation='sigmoid')
        )
        self.interest_net = ValueNetwork(
            name='interest_net',
            value_net_type=OutputNetworkType.CRITIC_QVALUE_ALL,
            value_net_kwargs=dict(vector_dim=self._representation_net.h_dim,
                                  output_shape=self.options_num,
                                  network_settings=network_settings['interest'],
                                  out_activation='sigmoid')
        )

        self.actor_tv = self.intra_option_net.trainable_variables
        if self.is_continuous:
            self.log_std = tf.Variable(initial_value=-0.5 * np.ones((self.options_num, self.a_dim), dtype=np.float32), trainable=True)   # [P, A]
            self.actor_tv += [self.log_std]
        update_target_net_weights(self.q_target_net.weights, self.q_net.weights)

        self.q_lr, self.intra_option_lr, self.termination_lr, self.interest_lr = map(self.init_lr, [q_lr, intra_option_lr, termination_lr, interest_lr])
        self.q_optimizer = self.init_optimizer(self.q_lr, clipvalue=5.)
        self.intra_option_optimizer = self.init_optimizer(self.intra_option_lr, clipvalue=5.)
        self.termination_optimizer = self.init_optimizer(self.termination_lr, clipvalue=5.)
        self.interest_optimizer = self.init_optimizer(self.interest_lr, clipvalue=5.)

        self._worker_params_dict.update(self.q_net._policy_models)
        self._worker_params_dict.update(self.intra_option_net._policy_models)
        self._worker_params_dict.update(self.interest_net._policy_models)

        self._all_params_dict.update(self.q_net._all_models)
        self._all_params_dict.update(self.intra_option_net._all_models)
        self._all_params_dict.update(self.interest_net._all_models)
        self._all_params_dict.update(self.termination_net._all_models)
        self._all_params_dict.update(q_optimizer=self.q_optimizer,
                                     intra_option_optimizer=self.intra_option_optimizer,
                                     termination_optimizer=self.termination_optimizer,
                                     interest_optimizer=self.interest_optimizer)
        self._model_post_process()

    def _generate_random_options(self):
        return tf.constant(np.random.randint(0, self.options_num, self.n_agents), dtype=tf.int32)

    def choose_action(self, s, visual_s, evaluation=False):
        if not hasattr(self, 'options'):
            self.options = self._generate_random_options()
        self.last_options = self.options

        a, self.options, self.cell_state = self._get_action(s, visual_s, self.cell_state, self.options)
        a = a.numpy()
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state, options):
        with tf.device(self.device):
            feat, cell_state = self._representation_net(s, visual_s, cell_state=cell_state)
            q = self.q_net.value_net(feat)  # [B, P]
            pi = self.intra_option_net.value_net(feat)  # [B, P, A]
            options_onehot = tf.one_hot(options, self.options_num, dtype=tf.float32)    # [B, P]
            options_onehot_expanded = tf.expand_dims(options_onehot, axis=-1)  # [B, P, 1]
            pi = tf.reduce_sum(pi * options_onehot_expanded, axis=1)  # [B, A]
            if self.is_continuous:
                log_std = tf.gather(self.log_std, options)
                mu = tf.math.tanh(pi)
                a, _ = gaussian_clip_rsample(mu, log_std)
            else:
                pi = pi / self.boltzmann_temperature
                dist = tfp.distributions.Categorical(logits=tf.nn.log_softmax(pi))  # [B, ]
                a = dist.sample()
            interests = self.interest_net.value_net(feat)  # [B, P]
            op_logits = interests * q  # [B, P] or tf.nn.softmax(q)
            new_options = tfp.distributions.Categorical(logits=tf.nn.log_softmax(op_logits)).sample()
        return a, new_options, cell_state

    def _target_params_update(self):
        if self.global_step % self.assign_interval == 0:
            update_target_net_weights(self.q_target_net.weights, self.q_net.weights)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'sample_data_list': ['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done', 'last_options', 'options'],
                'train_data_list': ['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done', 'last_options', 'options'],
                'summary_dict': dict([
                    ['LEARNING_RATE/q_lr', self.q_lr(self.train_step)],
                    ['LEARNING_RATE/intra_option_lr', self.intra_option_lr(self.train_step)],
                    ['LEARNING_RATE/termination_lr', self.termination_lr(self.train_step)],
                    ['Statistics/option', self.options[0]]
                ])
            })

    @tf.function(experimental_relax_shapes=True)
    def _train(self, memories, isw, cell_state):
        s, visual_s, a, r, s_, visual_s_, done, last_options, options = memories
        last_options = tf.cast(last_options, tf.int32)
        options = tf.cast(options, tf.int32)
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                feat, _ = self._representation_net(s, visual_s, cell_state=cell_state)
                feat_, _ = self._representation_target_net(s_, visual_s_, cell_state=cell_state)
                q = self.q_net.value_net(feat)  # [B, P]
                pi = self.intra_option_net.value_net(feat)  # [B, P, A]
                beta = self.termination_net.value_net(feat)   # [B, P]
                q_next = self.q_target_net.value_net(feat_)   # [B, P], [B, P, A], [B, P]
                beta_next = self.termination_net.value_net(feat_)  # [B, P]
                interests = self.interest_net.value_net(feat)  # [B, P]
                options_onehot = tf.one_hot(options, self.options_num, dtype=tf.float32)    # [B,] => [B, P]

                q_s = qu_eval = tf.reduce_sum(q * options_onehot, axis=-1, keepdims=True)  # [B, 1]
                beta_s_ = tf.reduce_sum(beta_next * options_onehot, axis=-1, keepdims=True)  # [B, 1]
                q_s_ = tf.reduce_sum(q_next * options_onehot, axis=-1, keepdims=True)   # [B, 1]
                if self.double_q:
                    q_ = self.q_net.value_net(feat)  # [B, P], [B, P, A], [B, P]
                    max_a_idx = tf.one_hot(tf.argmax(q_, axis=-1), self.options_num, dtype=tf.float32)  # [B, P] => [B, ] => [B, P]
                    q_s_max = tf.reduce_sum(q_next * max_a_idx, axis=-1, keepdims=True)   # [B, 1]
                else:
                    q_s_max = tf.reduce_max(q_next, axis=-1, keepdims=True)   # [B, 1]
                u_target = (1 - beta_s_) * q_s_ + beta_s_ * q_s_max   # [B, 1]
                qu_target = tf.stop_gradient(r + self.gamma * (1 - done) * u_target)
                td_error = qu_target - qu_eval     # gradient : q
                q_loss = tf.reduce_mean(tf.square(td_error) * isw)        # [B, 1] => 1

                if self.use_baseline:
                    adv = tf.stop_gradient(qu_target - qu_eval)
                else:
                    adv = tf.stop_gradient(qu_target)
                options_onehot_expanded = tf.expand_dims(options_onehot, axis=-1)   # [B, P] => [B, P, 1]
                pi = tf.reduce_sum(pi * options_onehot_expanded, axis=1)  # [B, P, A] => [B, A]
                if self.is_continuous:
                    log_std = tf.gather(self.log_std, options)
                    mu = tf.math.tanh(pi)
                    log_p = gaussian_likelihood_sum(a, mu, log_std)
                    entropy = gaussian_entropy(log_std)
                else:
                    pi = pi / self.boltzmann_temperature
                    log_pi = tf.nn.log_softmax(pi, axis=-1)  # [B, A]
                    entropy = -tf.reduce_sum(tf.exp(log_pi) * log_pi, axis=1, keepdims=True)    # [B, 1]
                    log_p = tf.reduce_sum(a * log_pi, axis=-1, keepdims=True)   # [B, 1]
                pi_loss = tf.reduce_mean(-(log_p * adv + self.ent_coff * entropy))              # [B, 1] * [B, 1] => [B, 1] => 1

                last_options_onehot = tf.one_hot(last_options, self.options_num, dtype=tf.float32)    # [B,] => [B, P]
                beta_s = tf.reduce_sum(beta * last_options_onehot, axis=-1, keepdims=True)   # [B, 1]

                pi_op = tf.nn.softmax(interests * tf.stop_gradient(q))  # [B, P] or tf.nn.softmax(q)
                interest_loss = -tf.reduce_mean(beta_s * tf.reduce_sum(pi_op * options_onehot, axis=-1, keepdims=True) * q_s)  # [B, 1] => 1

                v_s = tf.reduce_sum(q * pi_op, axis=-1, keepdims=True)  # [B, P] * [B, P] => [B, 1]
                beta_loss = beta_s * tf.stop_gradient(q_s - v_s)   # [B, 1]
                if self.terminal_mask:
                    beta_loss *= (1 - done)
                beta_loss = tf.reduce_mean(beta_loss)  # [B, 1] => 1

            q_grads = tape.gradient(q_loss, self.q_net.trainable_variables)
            intra_option_grads = tape.gradient(pi_loss, self.actor_tv)
            termination_grads = tape.gradient(beta_loss, self.termination_net.trainable_variables)
            interest_grads = tape.gradient(interest_loss, self.interest_net.trainable_variables)
            self.q_optimizer.apply_gradients(
                zip(q_grads, self.q_net.trainable_variables)
            )
            self.intra_option_optimizer.apply_gradients(
                zip(intra_option_grads, self.actor_tv)
            )
            self.termination_optimizer.apply_gradients(
                zip(termination_grads, self.termination_net.trainable_variables)
            )
            self.interest_optimizer.apply_gradients(
                zip(interest_grads, self.interest_net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/q_loss', tf.reduce_mean(q_loss)],
                ['LOSS/pi_loss', tf.reduce_mean(pi_loss)],
                ['LOSS/beta_loss', tf.reduce_mean(beta_loss)],
                ['LOSS/interest_loss', tf.reduce_mean(interest_loss)],
                ['Statistics/q_option_max', tf.reduce_max(q_s)],
                ['Statistics/q_option_min', tf.reduce_min(q_s)],
                ['Statistics/q_option_mean', tf.reduce_mean(q_s)]
            ])

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        """
        for off-policy training, use this function to store <s, a, r, s_, done> into ReplayBuffer.
        """
        assert isinstance(a, np.ndarray), "store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store need done type is np.ndarray"
        self._running_average(s)
        self.data.add(
            s,
            visual_s,
            a,
            r[:, np.newaxis],   # 升维
            s_,
            visual_s_,
            done[:, np.newaxis],  # 升维
            self.last_options,
            self.options
        )

    def no_op_store(self, s, visual_s, a, r, s_, visual_s_, done):
        pass
