#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from dataclasses import dataclass

from rls.utils.tf2_utils import (gaussian_clip_rsample,
                                 gaussian_likelihood_sum,
                                 gaussian_entropy)
from rls.algos.base.on_policy import On_Policy
from rls.utils.build_networks import ValueNetwork
from rls.utils.specs import (OutputNetworkType,
                             ModelObservations,
                             RlsDataClass,
                             BatchExperiences)


@dataclass
class PPOC_Store_BatchExperiences(BatchExperiences):
    value: np.ndarray
    log_prob: np.ndarray
    o_log_prob: np.ndarray
    beta_advantage: np.ndarray
    last_options: np.ndarray
    options: np.ndarray


@dataclass
class PPOC_Train_BatchExperiences(RlsDataClass):
    obs: ModelObservations
    action: np.ndarray
    value: np.ndarray
    log_prob: np.ndarray
    o_log_prob: np.ndarray
    discounted_reward: np.ndarray
    gae_adv: np.ndarray
    beta_advantage: np.ndarray
    last_options: np.ndarray
    options: np.ndarray


class PPOC(On_Policy):
    '''
    Learnings Options End-to-End for Continuous Action Tasks, PPOC, http://arxiv.org/abs/1712.00004
    '''

    def __init__(self,
                 envspec,

                 options_num=4,
                 dc=0.01,
                 terminal_mask=False,
                 o_beta=1.0e-3,
                 epoch=4,
                 pi_beta=1.0e-3,
                 lr=5.0e-4,
                 lambda_=0.95,
                 epsilon=0.2,
                 value_epsilon=0.2,
                 kl_reverse=False,
                 kl_target=0.02,
                 kl_target_cutoff=2,
                 kl_target_earlystop=4,
                 kl_beta=[0.7, 1.3],
                 kl_alpha=1.5,
                 kl_coef=1.0,
                 network_settings={
                     'share': [32, 32],
                     'q': [32, 32],
                     'intra_option': [32, 32],
                     'termination': [32, 32],
                     'o': [32, 32]
                 },
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.pi_beta = pi_beta
        self.epoch = epoch
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.value_epsilon = value_epsilon
        self.kl_reverse = kl_reverse
        self.kl_target = kl_target
        self.kl_alpha = kl_alpha
        self.kl_coef = tf.constant(kl_coef, dtype=tf.float32)

        self.kl_cutoff = kl_target * kl_target_cutoff
        self.kl_stop = kl_target * kl_target_earlystop
        self.kl_low = kl_target * kl_beta[0]
        self.kl_high = kl_target * kl_beta[-1]

        self.options_num = options_num
        self.dc = dc
        self.terminal_mask = terminal_mask
        self.o_beta = o_beta

        self.net = ValueNetwork(
            name='net',
            representation_net=self._representation_net,
            value_net_type=OutputNetworkType.PPOC_SHARE,
            value_net_kwargs=dict(action_dim=self.a_dim,
                                  options_num=self.options_num,
                                  network_settings=network_settings, is_continuous=self.is_continuous)
        )

        if self.is_continuous:
            self.log_std = tf.Variable(initial_value=-0.5 * np.ones((self.options_num, self.a_dim), dtype=np.float32), trainable=True)   # [P, A]
            self.net_tv = self.net.trainable_variables + [self.log_std]
        else:
            self.net_tv = self.net.trainable_variables
        self.lr = self.init_lr(lr)
        self.optimizer = self.init_optimizer(self.lr)

        self.initialize_data_buffer(store_data_type=PPOC_Store_BatchExperiences,
                                    sample_data_type=PPOC_Train_BatchExperiences)

        self._worker_params_dict.update(self.net._policy_models)

        self._all_params_dict.update(self.net._all_models)
        self._all_params_dict.update(optimizer=self.optimizer)
        self._model_post_process()

    def reset(self):
        super().reset()
        self._done_mask = np.full(self.n_copys, True)

    def partial_reset(self, done):
        super().partial_reset(done)
        self._done_mask = done

    def _generate_random_options(self):
        return tf.constant(np.random.randint(0, self.options_num, self.n_copys), dtype=tf.int32)

    def choose_action(self, obs, evaluation=False):
        if not hasattr(self, 'options'):
            self.options = self._generate_random_options()
        self.last_options = self.options
        if not hasattr(self, 'oc_mask'):
            self.oc_mask = tf.constant(np.zeros(self.n_copys), dtype=tf.int32)

        a, value, log_prob, o_log_prob, beta_adv, new_options, max_options, self.next_cell_state = self._get_action(obs.nt, self.cell_state, self.options)
        a = a.numpy()
        new_options = tf.where(self._done_mask, max_options, new_options)
        self._done_mask = np.full(self.n_copys, False)
        self._value = value.numpy()
        self._log_prob = log_prob.numpy() + 1e-10
        self._o_log_prob = o_log_prob.numpy() + 1e-10
        self._beta_adv = beta_adv.numpy() + self.dc
        self.oc_mask = (new_options == self.options).numpy()  # equal means no change
        self.options = new_options
        return a

    @tf.function
    def _get_action(self, obs, cell_state, options):
        with tf.device(self.device):
            ret = self.net(obs, cell_state=cell_state)  # [B, P], [B, P, A], [B, P], [B, P]
            (q, pi, beta, o) = ret['value']
            options_onehot = tf.one_hot(options, self.options_num, dtype=tf.float32)    # [B, P]
            options_onehot_expanded = tf.expand_dims(options_onehot, axis=-1)  # [B, P, 1]
            pi = tf.reduce_sum(pi * options_onehot_expanded, axis=1)  # [B, A]
            if self.is_continuous:
                log_std = tf.gather(self.log_std, options)
                mu = pi
                sample_op, _ = gaussian_clip_rsample(mu, log_std)
                log_prob = gaussian_likelihood_sum(sample_op, mu, log_std)
            else:
                logits = pi
                norm_dist = tfp.distributions.Categorical(logits=logits)
                sample_op = norm_dist.sample()
                log_prob = norm_dist.log_prob(sample_op)
            o_log_prob = tf.reduce_sum(o * options_onehot, axis=-1, keepdims=True)   # [B, 1]
            q_o = tf.reduce_sum(q * options_onehot, axis=-1, keepdims=True)  # [B, 1]
            beta_adv = q_o - tf.reduce_sum(q * tf.math.exp(o), axis=-1, keepdims=True)   # [B, 1]
            option_norm_dist = tfp.distributions.Categorical(logits=o)
            sample_options = option_norm_dist.sample()
            max_options = tf.cast(tf.argmax(q, axis=-1), dtype=tf.int32)  # [B, P] => [B, ]
            beta_probs = tf.reduce_sum(beta * options_onehot, axis=1)   # [B, P] => [B,]
            beta_dist = tfp.distributions.Bernoulli(probs=beta_probs)
            new_options = tf.where(beta_dist.sample() < 1, options, sample_options)    # <1 则不改变op， =1 则改变op
        return sample_op, q_o, log_prob, o_log_prob, beta_adv, new_options, max_options, ret['cell_state']

    def store_data(self, exps: BatchExperiences):
        # self._running_average()
        exps.reward = exps.reward - tf.expand_dims((1 - self.oc_mask) * self.dc, axis=-1)
        self.data.add(PPOC_Store_BatchExperiences(*exps, self._value, self._log_prob, self._o_log_prob, self._beta_adv,
                                                  self.last_options, self.options))
        if self.use_rnn:
            self.data.add_cell_state(tuple(cs.numpy() for cs in self.cell_state))
        self.cell_state = self.next_cell_state
        self.oc_mask = tf.zeros_like(self.oc_mask)

    @tf.function
    def _get_value(self, obs, options, cell_state):
        options = tf.cast(options, tf.int32)
        with tf.device(self.device):
            ret = self.net(obs, cell_state=cell_state)
            (q, _, _, _) = ret['value']
            options_onehot = tf.one_hot(options, self.options_num, dtype=tf.float32)    # [B, P]
            value = q_o = tf.reduce_sum(q * options_onehot, axis=-1, keepdims=True)  # [B, 1]
            return value, ret['cell_state']

    def calculate_statistics(self):
        init_value, self.cell_state = self._get_value(self.data.last_data('obs_'), self.data.last_data('options'), cell_state=self.cell_state)
        init_value = init_value.numpy()
        self.data.cal_dc_r(self.gamma, init_value)
        self.data.cal_td_error(self.gamma, init_value)
        self.data.cal_gae_adv(self.lambda_, self.gamma)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        def _train(data, cell_state):
            early_step = 0
            for i in range(self.epoch):
                loss, pi_loss, q_loss, o_loss, beta_loss, entropy, o_entropy, kl = self.share(data, cell_state, self.kl_coef)
                if kl > self.kl_stop:
                    early_step = i
                    break

            if kl > self.kl_high:
                self.kl_coef *= self.kl_alpha
            elif kl < self.kl_low:
                self.kl_coef /= self.kl_alpha

            summaries = dict([
                ['LOSS/loss', loss],
                ['LOSS/loss', pi_loss],
                ['LOSS/loss', q_loss],
                ['LOSS/loss', o_loss],
                ['LOSS/loss', beta_loss],
                ['Statistics/kl', kl],
                ['Statistics/entropy', entropy],
                ['Statistics/o_entropy', o_entropy],
                ['Statistics/kl_coef', self.kl_coef],
                ['Statistics/early_step', early_step],
            ])
            return summaries

        summary_dict = dict([['LEARNING_RATE/lr', self.lr(self.train_step)]])

        self._learn(function_dict={
            'calculate_statistics': self.calculate_statistics,
            'train_function': _train,
            'summary_dict': summary_dict
        })

    @tf.function
    def share(self, BATCH, cell_state, kl_coef):
        last_options = tf.cast(BATCH.last_options, tf.int32)  # [B,]
        options = tf.cast(BATCH.options, tf.int32)
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                ret = self.net(BATCH.obs, cell_state=cell_state)  # [B, P], [B, P, A], [B, P], [B, P]
                (q, pi, beta, o) = ret['value']
                options_onehot = tf.one_hot(options, self.options_num, dtype=tf.float32)    # [B, P]
                options_onehot_expanded = tf.expand_dims(options_onehot, axis=-1)  # [B, P, 1]
                last_options_onehot = tf.one_hot(last_options, self.options_num, dtype=tf.float32)    # [B,] => [B, P]

                pi = tf.reduce_sum(pi * options_onehot_expanded, axis=1)  # [B, P, A] => [B, A]
                value = tf.reduce_sum(q * options_onehot, axis=1, keepdims=True)    # [B, 1]

                if self.is_continuous:
                    log_std = tf.gather(self.log_std, options)
                    mu = pi  # [B, A]
                    new_log_prob = gaussian_likelihood_sum(BATCH.action, mu, log_std)
                    entropy = gaussian_entropy(log_std)
                else:
                    logits = pi  # [B, A]
                    logp_all = tf.nn.log_softmax(logits)
                    new_log_prob = tf.reduce_sum(BATCH.action * logp_all, axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                ratio = tf.exp(new_log_prob - BATCH.log_prob)

                if self.kl_reverse:
                    kl = tf.reduce_mean(new_log_prob - BATCH.log_prob)
                else:
                    kl = tf.reduce_mean(BATCH.log_prob - new_log_prob)    # a sample estimate for KL-divergence, easy to compute
                surrogate = ratio * BATCH.gae_adv

                value_clip = BATCH.value + tf.clip_by_value(value - BATCH.value, -self.value_epsilon, self.value_epsilon)
                td_error = BATCH.discounted_reward - value
                td_error_clip = BATCH.discounted_reward - value_clip
                td_square = tf.maximum(tf.square(td_error), tf.square(td_error_clip))

                pi_loss = -tf.reduce_mean(
                    tf.minimum(
                        surrogate,
                        tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * BATCH.gae_adv
                    ))
                kl_loss = kl_coef * kl
                extra_loss = 1000.0 * tf.square(tf.maximum(0., kl - self.kl_cutoff))
                pi_loss = pi_loss + kl_loss + extra_loss
                q_loss = 0.5 * tf.reduce_mean(td_square)

                beta_s = tf.reduce_sum(beta * last_options_onehot, axis=-1, keepdims=True)   # [B, 1]
                beta_loss = tf.reduce_mean(beta_s * BATCH.beta_advantage)
                if self.terminal_mask:
                    beta_loss *= (1 - done)

                o_log_prob = tf.reduce_sum(o * options_onehot, axis=-1, keepdims=True)   # [B, 1]
                o_ratio = tf.exp(o_log_prob - BATCH.o_log_prob)
                o_entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(o) * o, axis=1, keepdims=True))
                o_loss = -tf.reduce_mean(
                    tf.minimum(
                        o_ratio * BATCH.gae_adv,
                        tf.clip_by_value(o_ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * BATCH.gae_adv
                    ))

                loss = pi_loss + 1.0 * q_loss + o_loss + beta_loss - self.pi_beta * entropy - self.o_beta * o_entropy
            loss_grads = tape.gradient(loss, self.net_tv)
            self.optimizer.apply_gradients(
                zip(loss_grads, self.net_tv)
            )
            self.global_step.assign_add(1)
            return loss, pi_loss, q_loss, o_loss, beta_loss, entropy, o_entropy, kl
