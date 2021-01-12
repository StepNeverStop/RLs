#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from rls.algos.base.off_policy import Off_Policy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.tf2_utils import update_target_net_weights
from rls.utils.build_networks import DoubleValueNetwork
from rls.utils.specs import OutputNetworkType


class MAXSQN(Off_Policy):
    '''
    https://github.com/createamind/DRL/blob/master/spinup/algos/maxsqn/maxsqn.py
    '''

    def __init__(self,
                 envspec,

                 alpha=0.2,
                 beta=0.1,
                 ployak=0.995,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 use_epsilon=False,
                 q_lr=5.0e-4,
                 alpha_lr=5.0e-4,
                 auto_adaption=True,
                 network_settings=[32, 32],
                 **kwargs):
        assert not envspec.is_continuous, 'maxsqn only support discrete action space'
        super().__init__(envspec=envspec, **kwargs)
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.use_epsilon = use_epsilon
        self.ployak = ployak
        self.log_alpha = alpha if not auto_adaption else tf.Variable(initial_value=0.0, name='log_alpha', dtype=tf.float32, trainable=True)
        self.auto_adaption = auto_adaption
        self.target_entropy = beta * np.log(self.a_dim)

        def _create_net(name, representation_net=None): return DoubleValueNetwork(
            name=name,
            representation_net=representation_net,
            value_net_type=OutputNetworkType.CRITIC_QVALUE_ALL,
            value_net_kwargs=dict(output_shape=self.a_dim, network_settings=network_settings)
        )
        self.critic_net = _create_net('critic_net', self._representation_net)
        self._representation_target_net = self._create_representation_net('_representation_target_net')
        self.critic_target_net = _create_net('critic_target_net', self._representation_target_net)

        update_target_net_weights(self.critic_target_net.weights, self.critic_net.weights)
        self.q_lr, self.alpha_lr = map(self.init_lr, [q_lr, alpha_lr])
        self.optimizer_critic, self.optimizer_alpha = map(self.init_optimizer, [self.q_lr, self.alpha_lr])

        self._worker_params_dict.update(self.critic_net._policy_models)

        self._all_params_dict.update(self.critic_net._all_models)
        self._all_params_dict.update(optimizer_critic=self.optimizer_critic,
                                     optimizer_alpha=self.optimizer_alpha)
        self._model_post_process()

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def choose_action(self, obs, evaluation=False):
        if self.use_epsilon and np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
            a = np.random.randint(0, self.a_dim, self.n_agents)
        else:
            mu, pi, self.cell_state = self._get_action(obs, self.cell_state)
            a = pi.numpy()
        return a

    @tf.function
    def _get_action(self, obs, cell_state):
        with tf.device(self.device):
            q, _, cell_state = self.critic_net(obs, cell_state=cell_state)
            cate_dist = tfp.distributions.Categorical(logits=(q / self.alpha))
            pi = cate_dist.sample()
        return tf.argmax(q, axis=1), pi, cell_state

    def _target_params_update(self):
        update_target_net_weights(self.critic_target_net.weights, self.critic_net.weights, self.ployak)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([
                    ['LEARNING_RATE/q_lr', self.q_lr(self.train_step)],
                    ['LEARNING_RATE/alpha_lr', self.alpha_lr(self.train_step)]
                ])
            })

    @tf.function
    def _train(self, BATCH, isw, cell_state):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                q1, q2, _ = self.critic_net(BATCH.obs, cell_state=cell_state)
                q1_eval = tf.reduce_sum(tf.multiply(q1, BATCH.action), axis=1, keepdims=True)
                q2_eval = tf.reduce_sum(tf.multiply(q2, BATCH.action), axis=1, keepdims=True)

                q1_target, q2_target, _ = self.critic_target_net(BATCH.obs_, cell_state=cell_state)
                q1_target_max = tf.reduce_max(q1_target, axis=1, keepdims=True)
                q1_target_log_probs = tf.nn.log_softmax(q1_target / (self.alpha + 1e-8), axis=1)
                q1_target_entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(q1_target_log_probs) * q1_target_log_probs, axis=1, keepdims=True))

                q2_target_max = tf.reduce_max(q2_target, axis=1, keepdims=True)
                # q2_target_log_probs = tf.nn.log_softmax(q2_target, axis=1)
                # q2_target_log_max = tf.reduce_max(q2_target_log_probs, axis=1, keepdims=True)

                q_target = tf.minimum(q1_target_max, q2_target_max) + self.alpha * q1_target_entropy
                dc_r = tf.stop_gradient(BATCH.reward + self.gamma * q_target * (1 - BATCH.done))
                td_error1 = q1_eval - dc_r
                td_error2 = q2_eval - dc_r
                q1_loss = tf.reduce_mean(tf.square(td_error1) * isw)
                q2_loss = tf.reduce_mean(tf.square(td_error2) * isw)
                loss = 0.5 * (q1_loss + q2_loss)
                if self.auto_adaption:
                    q1_log_probs = tf.nn.log_softmax(q1 / (self.alpha + 1e-8), axis=1)
                    q1_entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(q1_log_probs) * q1_log_probs, axis=1, keepdims=True))
                    alpha_loss = -tf.reduce_mean(self.alpha * tf.stop_gradient(self.target_entropy - q1_entropy))
            loss_grads = tape.gradient(loss, self.critic_net.trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(loss_grads, self.critic_net.trainable_variables)
            )
            if self.auto_adaption:
                alpha_grad = tape.gradient(alpha_loss, self.log_alpha)
                self.optimizer_alpha.apply_gradients(
                    [(alpha_grad, self.log_alpha)]
                )
            self.global_step.assign_add(1)
            summaries = dict([
                ['LOSS/loss', loss],
                ['Statistics/log_alpha', self.log_alpha],
                ['Statistics/alpha', self.alpha],
                ['Statistics/q1_entropy', q1_entropy],
                ['Statistics/q_min', tf.reduce_mean(tf.minimum(q1, q2))],
                ['Statistics/q_mean', tf.reduce_mean(q1)],
                ['Statistics/q_max', tf.reduce_mean(tf.maximum(q1, q2))]
            ])
            if self.auto_adaption:
                summaries.update({
                    'LOSS/alpha_loss': alpha_loss
                })
            return (td_error1 + td_error2) / 2, summaries
