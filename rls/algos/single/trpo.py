#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from collections import namedtuple

from rls.utils.tf2_utils import (gaussian_clip_rsample,
                                 gaussian_likelihood_sum,
                                 gaussian_entropy)
from rls.algos.base.on_policy import On_Policy
from rls.utils.build_networks import ACNetwork
from rls.utils.specs import (OutputNetworkType,
                             BatchExperiences)

TRPO_Store_BatchExperiences_CTS = namedtuple('TRPO_Store_BatchExperiences_CTS', BatchExperiences._fields + ('value', 'log_prob', 'mu', 'log_std'))
TRPO_Store_BatchExperiences_DCT = namedtuple('TRPO_Store_BatchExperiences_DCT', BatchExperiences._fields + ('value', 'log_prob', 'logp_all'))
TRPO_Train_BatchExperiences_CTS = namedtuple('TRPO_Train_BatchExperiences_CTS', 'obs, action, log_prob, discounted_reward, gae_adv, mu, log_std')
TRPO_Train_BatchExperiences_DCT = namedtuple('TRPO_Train_BatchExperiences_DCT', 'obs, action, log_prob, discounted_reward, gae_adv, logp_all')


'''
Stole this from OpenAI SpinningUp. https://github.com/openai/spinningup/blob/master/spinup/algos/trpo/trpo.py
'''


def flat_concat(xs):
    return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)


def assign_params_from_flat(x, params):
    def flat_size(p): return int(np.prod(p.shape.as_list()))  # the 'int' is important for scalars
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([p.assign(p_new) for p, p_new in zip(params, new_params)])


class TRPO(On_Policy):
    '''
    Trust Region Policy Optimization, https://arxiv.org/abs/1502.05477
    '''

    def __init__(self,
                 envspec,

                 beta=1.0e-3,
                 lr=5.0e-4,
                 delta=0.01,
                 lambda_=0.95,
                 cg_iters=10,
                 train_v_iters=10,
                 damping_coeff=0.1,
                 backtrack_iters=10,
                 backtrack_coeff=0.8,
                 epsilon=0.2,
                 critic_lr=1e-3,
                 condition_sigma: bool = False,
                 network_settings={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'critic': [32, 32]
                 },
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.beta = beta
        self.delta = delta
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.cg_iters = cg_iters
        self.damping_coeff = damping_coeff
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.train_v_iters = train_v_iters

        if self.is_continuous:
            self.net = ACNetwork(
                name='net',
                representation_net=self._representation_net,
                policy_net_type=OutputNetworkType.ACTOR_MU_LOGSTD,
                policy_net_kwargs=dict(output_shape=self.a_dim,
                                       condition_sigma=condition_sigma,
                                       network_settings=network_settings['actor_continuous']),
                value_net_type=OutputNetworkType.CRITIC_VALUE,
                value_net_kwargs=dict(network_settings=network_settings['critic'])
            )
        else:
            self.net = ACNetwork(
                name='net',
                representation_net=self._representation_net,
                policy_net_type=OutputNetworkType.ACTOR_DCT,
                policy_net_kwargs=dict(output_shape=self.a_dim,
                                       network_settings=network_settings['actor_discrete']),
                value_net_type=OutputNetworkType.CRITIC_VALUE,
                value_net_kwargs=dict(network_settings=network_settings['critic'])
            )

        self.critic_lr = self.init_lr(critic_lr)
        self.optimizer_critic = self.init_optimizer(self.critic_lr)

        if self.is_continuous:
            self.initialize_data_buffer(store_data_type=TRPO_Store_BatchExperiences_CTS,
                                        sample_data_type=TRPO_Train_BatchExperiences_CTS)
        else:
            self.initialize_data_buffer(store_data_type=TRPO_Store_BatchExperiences_DCT,
                                        sample_data_type=TRPO_Train_BatchExperiences_DCT)

        self._worker_params_dict.update(self.net._policy_models)

        self._all_params_dict.update(self.net._all_models)
        self._all_params_dict.update(optimizer_critic=self.optimizer_critic)
        self._model_post_process()

    def choose_action(self, obs, evaluation=False):
        a, _v, _lp, _morlpa, self.next_cell_state = self._get_action(obs, self.cell_state)
        a = a.numpy()
        self._value = _v.numpy()
        self._log_prob = _lp.numpy() + 1e-10
        if self.is_continuous:
            self._mu = _morlpa[0].numpy()
            self._log_std = _morlpa[1].numpy()
        else:
            self._logp_all = _morlpa.numpy()
        return a

    @tf.function
    def _get_action(self, obs, cell_state):
        with tf.device(self.device):
            feat, cell_state = self._representation_net(obs, cell_state=cell_state)
            value = self.net.value_net(feat)
            output = self.net.policy_net(feat)
            if self.is_continuous:
                mu, log_std = output
                sample_op, _ = gaussian_clip_rsample(mu, log_std)
                log_prob = gaussian_likelihood_sum(sample_op, mu, log_std)
                return sample_op, value, log_prob, (mu, log_std), cell_state
            else:
                logits = output
                logp_all = tf.nn.log_softmax(logits)
                norm_dist = tfp.distributions.Categorical(logits=logp_all)
                sample_op = norm_dist.sample()
                log_prob = norm_dist.log_prob(sample_op)
                return sample_op, value, log_prob, logp_all, cell_state

    def store_data(self, exps: BatchExperiences):
        self._running_average(exps.obs.vector)

        if self.is_continuous:
            self.data.add(TRPO_Store_BatchExperiences_CTS(*exps, self._value, self._log_prob, self._mu, self._log_std))
        else:
            self.data.add(TRPO_Store_BatchExperiences_DCT(*exps, self._value, self._log_prob, self._logp_all))
        if self.use_rnn:
            self.data.add_cell_state(tuple(cs.numpy() for cs in self.cell_state))
        self.cell_state = self.next_cell_state

    @tf.function
    def _get_value(self, obs, cell_state):
        with tf.device(self.device):
            feat, cell_state = self._representation_net(obs, cell_state=cell_state)
            value = self.net.value_net(feat)
            return value, cell_state

    def calculate_statistics(self):
        init_value, self.cell_state = self._get_value(self.data.last_data('obs_'), cell_state=self.cell_state)
        init_value = init_value.numpy()
        self.data.cal_dc_r(self.gamma, init_value)
        self.data.cal_td_error(self.gamma, init_value)
        self.data.cal_gae_adv(self.lambda_, self.gamma)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        def _train(data, cell_state):
            actor_loss, entropy, gradients = self.train_actor(data, cell_state)

            x = self.cg(self.Hx, gradients.numpy(), data, cell_state)
            x = tf.convert_to_tensor(x)
            alpha = np.sqrt(2 * self.delta / (np.dot(x, self.Hx(x, data, cell_state)) + 1e-8))
            for i in range(self.backtrack_iters):
                assign_params_from_flat(alpha * x * (self.backtrack_coeff ** i), self.net.actor_trainable_variables)

            for _ in range(self.train_v_iters):
                critic_loss = self.train_critic(data, cell_state)

            summaries = dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/entropy', entropy]
            ])
            return summaries

        self._learn(function_dict={
            'calculate_statistics': self.calculate_statistics,
            'train_function': _train,
            'summary_dict': dict([
                ['LEARNING_RATE/critic_lr', self.critic_lr(self.train_step)]
            ])
        })

    @tf.function
    def train_actor(self, BATCH, cell_state):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                output, _ = self.net(BATCH.obs, cell_state=cell_state)
                if self.is_continuous:
                    mu, log_std = output
                    new_log_prob = gaussian_likelihood_sum(BATCH.action, mu, log_std)
                    entropy = gaussian_entropy(log_std)
                else:
                    logits = output
                    logp_all = tf.nn.log_softmax(logits)
                    new_log_prob = tf.reduce_sum(BATCH.action * logp_all, axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                ratio = tf.exp(new_log_prob - BATCH.log_prob)
                actor_loss = -tf.reduce_mean(ratio * BATCH.gae_adv)
            actor_grads = tape.gradient(actor_loss, self.net.actor_trainable_variables)
            gradients = flat_concat(actor_grads)
            self.global_step.assign_add(1)
            return actor_loss, entropy, gradients

    @tf.function
    def Hx(self, x, BATCH, cell_state):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                output, _ = self.net(BATCH.obs, cell_state=cell_state)
                if self.is_continuous:
                    mu, log_std = output
                    var0, var1 = tf.exp(2 * log_std), tf.exp(2 * BATCH.log_std)
                    pre_sum = 0.5 * (((BATCH.mu - mu)**2 + var0) / (var1 + 1e-8) - 1) + BATCH.log_std - log_std
                    all_kls = tf.reduce_sum(pre_sum, axis=1)
                    kl = tf.reduce_mean(all_kls)
                else:
                    logits = output
                    logp_all = tf.nn.log_softmax(logits)
                    all_kls = tf.reduce_sum(tf.exp(BATCH.logp_all) * (BATCH.logp_all - logp_all), axis=1)
                    kl = tf.reduce_mean(all_kls)

                g = flat_concat(tape.gradient(kl, self.net.actor_trainable_variables))
                _g = tf.reduce_sum(g * x)
            hvp = flat_concat(tape.gradient(_g, self.net.actor_trainable_variables))
            if self.damping_coeff > 0:
                hvp += self.damping_coeff * x
            return hvp

    @tf.function
    def train_critic(self, BATCH, cell_state):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat, _ = self._representation_net(BATCH.obs, cell_state=cell_state)
                value = self.net.value_net(feat)
                td_error = BATCH.discounted_reward - value
                value_loss = tf.reduce_mean(tf.square(td_error))
            critic_grads = tape.gradient(value_loss, self.net.critic_trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.net.critic_trainable_variables)
            )
            return value_loss

    def cg(self, Ax, b, BATCH, cell_state):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """
        x = np.zeros_like(b)
        r = b.copy()  # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = r.copy()
        r_dot_old = np.dot(r, r)
        for _ in range(self.cg_iters):
            z = Ax(tf.convert_to_tensor(p), BATCH, cell_state)
            alpha = r_dot_old / (np.dot(p, z) + 1e-8)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x
