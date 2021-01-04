#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import (Union,
                    List,
                    Dict,
                    NoReturn)
from collections import namedtuple

from rls.utils.tf2_utils import (show_graph,
                                 gaussian_clip_rsample,
                                 gaussian_likelihood_sum,
                                 gaussian_entropy)
from rls.algos.base.on_policy import On_Policy
from rls.utils.build_networks import (ValueNetwork,
                                      ACNetwork)
from rls.utils.specs import (OutputNetworkType,
                             ModelObservations,
                             BatchExperiences)

PPO_Store_BatchExperiences = namedtuple('PPO_Store_BatchExperiences', BatchExperiences._fields + ('value', 'log_prob'))
PPO_Train_BatchExperiences = namedtuple('PPO_Train_BatchExperiences', 'obs, action, value, log_prob, discounted_reward, gae_adv')


class PPO(On_Policy):
    '''
    Proximal Policy Optimization, https://arxiv.org/abs/1707.06347
    Emergence of Locomotion Behaviours in Rich Environments, http://arxiv.org/abs/1707.02286, DPPO
    '''

    def __init__(self,
                 envspec,

                 policy_epoch: int = 4,
                 value_epoch: int = 4,
                 ent_coef: float = 1.0e-2,
                 vf_coef: float = 0.5,
                 lr: float = 5.0e-4,
                 lambda_: float = 0.95,
                 epsilon: float = 0.2,
                 use_duel_clip: bool = False,
                 duel_epsilon: float = 0.,
                 use_vclip: bool = False,
                 value_epsilon: float = 0.2,
                 share_net: bool = True,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 1e-3,
                 max_grad_norm: float = 0.5,
                 condition_sigma: bool = False,
                 kl_reverse: bool = False,
                 kl_target: float = 0.02,
                 kl_target_cutoff: float = 2,
                 kl_target_earlystop: float = 4,
                 kl_beta: List[float] = [0.7, 1.3],
                 kl_alpha: float = 1.5,
                 kl_coef: float = 1.0,
                 extra_coef: float = 1000.0,
                 use_kl_loss: bool = False,
                 use_extra_loss: bool = False,
                 use_early_stop: bool = False,
                 network_settings: Dict = {
                     'share': {
                         'continuous': {
                             'share': [32, 32],
                             'mu': [32, 32],
                             'v': [32, 32]
                         },
                         'discrete': {
                             'share': [32, 32],
                             'logits': [32, 32],
                             'v': [32, 32]
                         }
                     },
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'critic': [32, 32]
                 },
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.ent_coef = ent_coef
        self.policy_epoch = policy_epoch
        self.value_epoch = value_epoch
        self.lambda_ = lambda_
        assert 0.0 <= lambda_ <= 1.0, "GAE lambda should be in [0, 1]."
        self.epsilon = epsilon
        self.use_vclip = use_vclip
        self.value_epsilon = value_epsilon
        self.share_net = share_net
        self.kl_reverse = kl_reverse
        self.kl_target = kl_target
        self.kl_alpha = kl_alpha
        self.kl_coef = tf.constant(kl_coef, dtype=tf.float32)
        self.extra_coef = extra_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.use_duel_clip = use_duel_clip
        self.duel_epsilon = duel_epsilon
        if self.use_duel_clip:
            assert -self.epsilon < self.duel_epsilon < self.epsilon, "duel_epsilon should be set in the range of (-epsilon, epsilon)."

        self.kl_cutoff = kl_target * kl_target_cutoff
        self.kl_stop = kl_target * kl_target_earlystop
        self.kl_low = kl_target * kl_beta[0]
        self.kl_high = kl_target * kl_beta[-1]

        self.use_kl_loss = use_kl_loss
        self.use_extra_loss = use_extra_loss
        self.use_early_stop = use_early_stop

        if self.share_net:
            if self.is_continuous:
                self.net = ValueNetwork(
                    name='net',
                    representation_net=self._representation_net,
                    value_net_type=OutputNetworkType.ACTOR_CRITIC_VALUE_CTS,
                    value_net_kwargs=dict(output_shape=self.a_dim,
                                          condition_sigma=condition_sigma,
                                          network_settings=network_settings['share']['continuous'])
                )
            else:
                self.net = ValueNetwork(
                    name='net',
                    representation_net=self._representation_net,
                    value_net_type=OutputNetworkType.ACTOR_CRITIC_VALUE_DET,
                    value_net_kwargs=dict(output_shape=self.a_dim,
                                          network_settings=network_settings['share']['discrete'])
                )
            self.lr = self.init_lr(lr)
            if self.max_grad_norm is not None:
                self.optimizer = self.init_optimizer(self.lr, clipnorm=self.max_grad_norm)
            else:
                self.optimizer = self.init_optimizer(self.lr)
            self._all_params_dict.update(optimizer=self.optimizer)
        else:
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
            self.actor_lr, self.critic_lr = map(self.init_lr, [actor_lr, critic_lr])
            if self.max_grad_norm is not None:
                self.optimizer_actor = self.init_optimizer(self.actor_lr, clipnorm=self.max_grad_norm)
                self.optimizer_critic = self.init_optimizer(self.critic_lr, clipnorm=self.max_grad_norm)
            else:
                self.optimizer_actor, self.optimizer_critic = map(self.init_optimizer, [self.actor_lr, self.critic_lr])

            self._all_params_dict.update(optimizer_actor=self.optimizer_actor,
                                         optimizer_critic=self.optimizer_critic)

        self._worker_params_dict.update(self.net._policy_models)

        self._all_params_dict.update(self.net._all_models)

        self.initialize_data_buffer(store_data_type=PPO_Store_BatchExperiences,
                                    sample_data_type=PPO_Train_BatchExperiences)
        self._model_post_process()

    def choose_action(self, obs: ModelObservations, evaluation: bool = False) -> np.ndarray:
        a, value, log_prob, self.next_cell_state = self._get_action(obs, self.cell_state)
        a = a.numpy()
        self._value = value.numpy()
        self._log_prob = log_prob.numpy() + 1e-10
        return a

    @tf.function
    def _get_action(self, obs, cell_state):
        with tf.device(self.device):
            feat, cell_state = self._representation_net(obs, cell_state=cell_state)
            if self.is_continuous:
                if self.share_net:
                    mu, log_std, value = self.net.value_net(feat)
                else:
                    mu, log_std = self.net.policy_net(feat)
                    value = self.net.value_net(feat)
                sample_op, _ = gaussian_clip_rsample(mu, log_std)
                log_prob = gaussian_likelihood_sum(sample_op, mu, log_std)
            else:
                if self.share_net:
                    logits, value = self.net.value_net(feat)
                else:
                    logits = self.net.policy_net(feat)
                    value = self.net.value_net(feat)
                norm_dist = tfp.distributions.Categorical(logits=tf.nn.log_softmax(logits))
                sample_op = norm_dist.sample()
                log_prob = norm_dist.log_prob(sample_op)
        return sample_op, value, log_prob, cell_state

    def store_data(self, exps: BatchExperiences) -> NoReturn:
        self._running_average(exps.obs.vector)
        self.data.add(PPO_Store_BatchExperiences(*exps, self._value, self._log_prob))
        if self.use_rnn:
            self.data.add_cell_state(tuple(cs.numpy() for cs in self.cell_state))
        self.cell_state = self.next_cell_state

    @tf.function
    def _get_value(self, obs, cell_state):
        with tf.device(self.device):
            feat, cell_state = self._representation_net(obs, cell_state=cell_state)
            output = self.net.value_net(feat)
            if self.is_continuous:
                if self.share_net:
                    _, _, value = output
                else:
                    value = output
            else:
                if self.share_net:
                    _, value = output
                else:
                    value = output
            return value, cell_state

    def calculate_statistics(self) -> NoReturn:
        init_value, self.cell_state = self._get_value(self.data.last_data('obs_'), cell_state=self.cell_state)
        init_value = init_value.numpy()
        self.data.cal_dc_r(self.gamma, init_value)
        self.data.cal_td_error(self.gamma, init_value)
        self.data.cal_gae_adv(self.lambda_, self.gamma, normalize=True)

    # @show_graph(name='ppo_net')
    def learn(self, **kwargs) -> NoReturn:
        self.train_step = kwargs.get('train_step')

        def _train(data, cell_state):
            early_step = 0
            if self.share_net:
                for i in range(self.policy_epoch):
                    actor_loss, critic_loss, entropy, kl = self.train_share(data, cell_state, self.kl_coef)
                    if self.use_early_stop and kl > self.kl_stop:
                        early_step = i
                        break
            else:
                for i in range(self.policy_epoch):
                    actor_loss, entropy, kl = self.train_actor(data, cell_state, self.kl_coef)
                    if self.use_early_stop and kl > self.kl_stop:
                        early_step = i
                        break

                for _ in range(self.value_epoch):
                    critic_loss = self.train_critic(data, cell_state)

            summaries = dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/kl', kl],
                ['Statistics/entropy', entropy]
            ])

            if self.use_early_stop:
                summaries.update(dict([
                    ['Statistics/early_step', early_step]
                ]))

            if self.use_kl_loss:
                # ref: https://github.com/joschu/modular_rl/blob/6970cde3da265cf2a98537250fea5e0c0d9a7639/modular_rl/ppo.py#L93
                if kl > self.kl_high:
                    self.kl_coef *= self.kl_alpha
                elif kl < self.kl_low:
                    self.kl_coef /= self.kl_alpha

                summaries.update(dict([
                    ['Statistics/kl_coef', self.kl_coef]
                ]))
            return summaries

        if self.share_net:
            summary_dict = dict([['LEARNING_RATE/lr', self.lr(self.train_step)]])
        else:
            summary_dict = dict([
                ['LEARNING_RATE/actor_lr', self.actor_lr(self.train_step)],
                ['LEARNING_RATE/critic_lr', self.critic_lr(self.train_step)]
            ])

        self._learn(function_dict={
            'calculate_statistics': self.calculate_statistics,
            'train_function': _train,
            'summary_dict': summary_dict,
            'train_data_type': PPO_Train_BatchExperiences
        })

    @tf.function(experimental_relax_shapes=True)
    def train_share(self, BATCH, cell_state, kl_coef):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                output, cell_state = self.net(BATCH.obs, cell_state=cell_state)
                if self.is_continuous:
                    mu, log_std, value = output
                    new_log_prob = gaussian_likelihood_sum(BATCH.action, mu, log_std)
                    entropy = gaussian_entropy(log_std)
                else:
                    logits, value = output
                    logp_all = tf.nn.log_softmax(logits)
                    new_log_prob = tf.reduce_sum(BATCH.action * logp_all, axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                ratio = tf.exp(new_log_prob - BATCH.log_prob)
                surrogate = ratio * BATCH.gae_adv
                clipped_surrogate = tf.minimum(
                    surrogate,
                    tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * BATCH.gae_adv
                )
                # ref: https://github.com/thu-ml/tianshou/blob/c97aa4065ee8464bd5897bb86f1f81abd8e2cff9/tianshou/policy/modelfree/ppo.py#L159
                if self.use_duel_clip:
                    clipped_surrogate = tf.maximum(
                        clipped_surrogate,
                        (1.0 + self.duel_epsilon) * BATCH.gae_adv
                    )
                actor_loss = -(tf.reduce_mean(clipped_surrogate) + self.ent_coef * entropy)

                # ref: https://github.com/joschu/modular_rl/blob/6970cde3da265cf2a98537250fea5e0c0d9a7639/modular_rl/ppo.py#L40
                # ref: https://github.com/hill-a/stable-baselines/blob/b3f414f4f2900403107357a2206f80868af16da3/stable_baselines/ppo2/ppo2.py#L185
                if self.kl_reverse:
                    kl = .5 * tf.reduce_mean(tf.square(new_log_prob - BATCH.log_prob))
                else:
                    kl = .5 * tf.reduce_mean(tf.square(BATCH.log_prob - new_log_prob))    # a sample estimate for KL-divergence, easy to compute

                td_error = BATCH.discounted_reward - value
                if self.use_vclip:
                    # ref: https://github.com/llSourcell/OpenAI_Five_vs_Dota2_Explained/blob/c5def7e57aa70785c2394ea2eeb3e5f66ad59a53/train.py#L154
                    # ref: https://github.com/hill-a/stable-baselines/blob/b3f414f4f2900403107357a2206f80868af16da3/stable_baselines/ppo2/ppo2.py#L172
                    value_clip = BATCH.value + tf.clip_by_value(value - BATCH.value, -self.value_epsilon, self.value_epsilon)
                    td_error_clip = BATCH.discounted_reward - value_clip
                    td_square = tf.maximum(tf.square(td_error), tf.square(td_error_clip))
                else:
                    td_square = tf.square(td_error)

                if self.use_kl_loss:
                    kl_loss = kl_coef * kl
                    actor_loss += kl_loss

                if self.use_extra_loss:
                    extra_loss = self.extra_coef * tf.square(tf.maximum(0., kl - self.kl_cutoff))
                    actor_loss += extra_loss
                value_loss = 0.5 * tf.reduce_mean(td_square)
                loss = actor_loss + self.vf_coef * value_loss
            loss_grads = tape.gradient(loss, self.net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(loss_grads, self.net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return actor_loss, value_loss, entropy, kl

    @tf.function(experimental_relax_shapes=True)
    def train_actor(self, BATCH, cell_state, kl_coef):
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
                kl = tf.reduce_mean(BATCH.log_prob - new_log_prob)
                surrogate = ratio * BATCH.gae_adv
                clipped_surrogate = tf.minimum(
                    surrogate,
                    tf.where(BATCH.gae_adv > 0, (1 + self.epsilon) * BATCH.gae_adv, (1 - self.epsilon) * BATCH.gae_adv)
                )
                if self.use_duel_clip:
                    clipped_surrogate = tf.maximum(
                        clipped_surrogate,
                        (1.0 + self.duel_epsilon) * BATCH.gae_adv
                    )

                actor_loss = -(tf.reduce_mean(clipped_surrogate) + self.ent_coef * entropy)

                if self.use_kl_loss:
                    kl_loss = kl_coef * kl
                    actor_loss += kl_loss
                if self.use_extra_loss:
                    extra_loss = self.extra_coef * tf.square(tf.maximum(0., kl - self.kl_cutoff))
                    actor_loss += extra_loss

            actor_grads = tape.gradient(actor_loss, self.net.actor_trainable_variables)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.net.actor_trainable_variables)
            )
            self.global_step.assign_add(1)
            return actor_loss, entropy, kl

    @tf.function(experimental_relax_shapes=True)
    def train_critic(self, BATCH, cell_state):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat, _ = self._representation_net(BATCH.obs, cell_state=cell_state)
                value = self.net.value_net(feat)

                td_error = BATCH.discounted_reward - value
                if self.use_vclip:
                    value_clip = BATCH.value + tf.clip_by_value(value - BATCH.value, -self.value_epsilon, self.value_epsilon)
                    td_error_clip = BATCH.discounted_reward - value_clip
                    td_square = tf.maximum(tf.square(td_error), tf.square(td_error_clip))
                else:
                    td_square = tf.square(td_error)

                value_loss = 0.5 * tf.reduce_mean(td_square)
            critic_grads = tape.gradient(value_loss, self.net.critic_trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.net.critic_trainable_variables)
            )
            return value_loss
