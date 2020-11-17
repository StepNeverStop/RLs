#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow.keras import Model as M
from tensorflow.keras import Input as I
from tensorflow.keras import Sequential
from skimage.util.shape import view_as_windows
from tensorflow.keras.layers import (Dense,
                                     Flatten,
                                     LayerNormalization)

from rls.utils.tf2_utils import (clip_nn_log_std,
                                 squash_rsample,
                                 gaussian_entropy,
                                 update_target_net_weights)
from rls.algos.base.off_policy import Off_Policy
from rls.utils.sundry_utils import LinearAnnealing
from rls.nn.networks import get_visual_network_from_type
from rls.nn.initializers import initKernelAndBias
from rls.utils.indexs import VisualNetworkType
from rls.utils.build_networks import (ValueNetwork,
                                      DoubleValueNetwork)
from rls.utils.indexs import OutputNetworkType


class VisualEncoder(M):

    def __init__(self, img_dim, fc_dim):
        super().__init__()
        self.net = Sequential([
            get_visual_network_from_type(VisualNetworkType.NATURE)(),
            Dense(fc_dim, **initKernelAndBias),
            LayerNormalization()
        ])
        self(I(shape=img_dim))

    def call(self, vis):
        return self.net(vis)


def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs


def center_crop_image(image, output_size):
    h, w = image.shape[1:3]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top:top + new_h, left:left + new_w]
    return image


class CURL(Off_Policy):
    """
    CURL: Contrastive Unsupervised Representations for Reinforcement Learning, http://arxiv.org/abs/2004.04136
    """

    def __init__(self,
                 envspec,

                 alpha=0.2,
                 annealing=True,
                 last_alpha=0.01,
                 ployak=0.995,
                 discrete_tau=1.0,
                 log_std_bound=[-20, 2],
                 network_settings={
                     'actor_continuous': {
                         'share': [128, 128],
                         'mu': [64],
                         'log_std': [64]
                     },
                     'actor_discrete': [64, 32],
                     'q': [128, 128],
                     'encoder': 128
                 },
                 auto_adaption=True,
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 alpha_lr=5.0e-4,
                 curl_lr=5.0e-4,
                 img_size=64,
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        assert self.visual_sources == 1
        self.ployak = ployak
        self.discrete_tau = discrete_tau
        self.log_std_min, self.log_std_max = log_std_bound[:]
        self.auto_adaption = auto_adaption
        self.annealing = annealing
        self.img_size = img_size
        self.img_dim = [img_size, img_size, self.visual_dim[-1]]
        self.vis_feat_size = network_settings['encoder']

        if self.auto_adaption:
            self.log_alpha = tf.Variable(initial_value=0.0, name='log_alpha', dtype=tf.float32, trainable=True)
        else:
            self.log_alpha = tf.Variable(initial_value=tf.math.log(alpha), name='log_alpha', dtype=tf.float32, trainable=False)
            if self.annealing:
                self.alpha_annealing = LinearAnnealing(alpha, last_alpha, 1.0e6)

        def _create_net(name): return DoubleValueNetwork(
            name=name,
            value_net_type=OutputNetworkType.CRITIC_QVALUE_ONE,
            value_net_kwargs=dict(vector_dim=self.s_dim + self.vis_feat_size,
                                  action_dim=self.a_dim,
                                  network_settings=network_settings['q'])
        )

        self.critic_net = _create_net('critic_net')
        self.critic_target_net = _create_net('critic_target_net')

        if self.is_continuous:
            self.actor_net = ValueNetwork(
                name='actor_net',
                value_net_type=OutputNetworkType.ACTOR_CTS,
                value_net_kwargs=dict(vector_dim=self.s_dim + self.vis_feat_size,
                                      output_shape=self.a_dim,
                                      network_settings=network_settings['actor_continuous'])
            )
        else:
            self.actor_net = ValueNetwork(
                name='actor_net',
                value_net_type=OutputNetworkType.ACTOR_DCT,
                value_net_kwargs=dict(vector_dim=self.s_dim + self.vis_feat_size,
                                      output_shape=self.a_dim,
                                      network_settings=network_settings['actor_discrete'])
            )
            self.gumbel_dist = tfp.distributions.Gumbel(0, 1)

        # entropy = -log(1/|A|) = log |A|
        self.target_entropy = 0.98 * (-self.a_dim if self.is_continuous else np.log(self.a_dim))

        self.encoder = VisualEncoder(self.img_dim, self.vis_feat_size)
        self.encoder_target = VisualEncoder(self.img_dim, self.vis_feat_size)

        self.curl_w = tf.Variable(initial_value=tf.random.normal(shape=(self.vis_feat_size, self.vis_feat_size)), name='curl_w', dtype=tf.float32, trainable=True)

        self.critic_tv = self.critic_net.trainable_variables + self.encoder.trainable_variables

        update_target_net_weights(
            self.critic_target_net.weights + self.encoder_target.trainable_variables,
            self.critic_net.weights + self.encoder.trainable_variables
        )
        self.actor_lr, self.critic_lr, self.alpha_lr, self.curl_lr = map(self.init_lr, [actor_lr, critic_lr, alpha_lr, curl_lr])
        self.optimizer_actor, self.optimizer_critic, self.optimizer_alpha, self.optimizer_curl = map(self.init_optimizer, [self.actor_lr, self.critic_lr, self.alpha_lr, self.curl_lr])

        self._worker_params_dict.update(self.actor_net._policy_models)
        self._worker_params_dict.update(encoder=self.encoder)

        self._all_params_dict.update(self.actor_net._all_models)
        self._all_params_dict.update(self.critic_net._all_models)
        self._all_params_dict.update(curl_w=self.curl_w,
                                     encoder=self.encoder,
                                     optimizer_actor=self.optimizer_actor,
                                     optimizer_critic=self.optimizer_critic,
                                     optimizer_alpha=self.optimizer_alpha,
                                     optimizer_curl=self.optimizer_curl)
        self._model_post_process()

    def choose_action(self, s, visual_s, evaluation=False):
        visual_s = center_crop_image(visual_s[:, 0], self.img_size)
        mu, pi = self._get_action(s, visual_s)
        a = mu.numpy() if evaluation else pi.numpy()
        return a

    @tf.function
    def _get_action(self, s, visual_s):
        with tf.device(self.device):
            feat = tf.concat([self.encoder(visual_s), s], axis=-1)
            if self.is_continuous:
                mu, log_std = self.actor_net.value_net(feat)
                log_std = clip_nn_log_std(log_std, self.log_std_min, self.log_std_max)
                pi, _ = squash_rsample(mu, log_std)
                mu = tf.tanh(mu)  # squash mu
            else:
                logits = self.actor_net.value_net(feat)
                mu = tf.argmax(logits, axis=1)
                cate_dist = tfp.distributions.Categorical(logits=tf.nn.log_softmax(logits))
                pi = cate_dist.sample()
            return mu, pi

    def _process_before_train(self, data):
        data['visual_s'] = np.transpose(data['visual_s'][:, 0].numpy(), (0, 3, 1, 2))
        data['visual_s_'] = np.transpose(data['visual_s_'][:, 0].numpy(), (0, 3, 1, 2))
        data['pos'] = self.data_convert(
            np.transpose(random_crop(data['visual_s'], self.img_size), (0, 2, 3, 1))
        )
        data['visual_s'] = self.data_convert(
            np.transpose(random_crop(data['visual_s'], self.img_size), (0, 2, 3, 1))
        )
        data['visual_s_'] = self.data_convert(
            np.transpose(random_crop(data['visual_s_'], self.img_size), (0, 2, 3, 1))
        )
        return (data,)

    def _target_params_update(self):
        update_target_net_weights(self.critic_target_net.weights + self.encoder_target.trainable_variables,
                                  self.critic_net.weights + self.encoder.trainable_variables,
                                  self.ployak)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([
                    ['LEARNING_RATE/actor_lr', self.actor_lr(self.train_step)],
                    ['LEARNING_RATE/critic_lr', self.critic_lr(self.train_step)],
                    ['LEARNING_RATE/alpha_lr', self.alpha_lr(self.train_step)]
                ]),
                'train_data_list': ['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done', 'pos'],
            })

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def _train(self, memories, isw, cell_state):
        td_error, summaries = self.train(memories, isw, cell_state)
        if self.annealing and not self.auto_adaption:
            self.log_alpha.assign(tf.math.log(tf.cast(self.alpha_annealing(self.global_step.numpy()), tf.float32)))
    return td_error, summaries

    @tf.function(experimental_relax_shapes=True)
    def train(self, memories, isw, cell_state):
        s, visual_s, a, r, s_, visual_s_, done, pos = memories
        batch_size = tf.shape(a)[0]
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                vis_feat = self.encoder(visual_s)
                vis_feat_ = self.encoder(visual_s_)
                target_vis_feat_ = self.encoder_target(visual_s_)
                feat = tf.concat([vis_feat, s], axis=-1)
                feat_ = tf.concat([vis_feat_, s_], axis=-1)
                target_feat_ = tf.concat([target_vis_feat_, s_], axis=-1)
                if self.is_continuous:
                    target_mu, target_log_std = self.actor_net.value_net(feat_)
                    target_log_std = clip_nn_log_std(target_log_std, self.log_std_min, self.log_std_max)
                    target_pi, target_log_pi = squash_rsample(target_mu, target_log_std)
                else:
                    target_logits = self.actor_net.value_net(feat_)
                    target_cate_dist = tfp.distributions.Categorical(logits=tf.nn.log_softmax(target_logits))
                    target_pi = target_cate_dist.sample()
                    target_log_pi = target_cate_dist.log_prob(target_pi)
                    target_pi = tf.one_hot(target_pi, self.a_dim, dtype=tf.float32)
                q1, q2 = self.critic_net.value_net(feat, a)
                q1_target, q2_target = self.critic_target_net.value_net(feat_, target_pi)
                dc_r_q1 = tf.stop_gradient(r + self.gamma * (1 - done) * (q1_target - self.alpha * target_log_pi))
                dc_r_q2 = tf.stop_gradient(r + self.gamma * (1 - done) * (q2_target - self.alpha * target_log_pi))
                td_error1 = q1 - dc_r_q1
                td_error2 = q2 - dc_r_q2
                q1_loss = tf.reduce_mean(tf.square(td_error1) * isw)
                q2_loss = tf.reduce_mean(tf.square(td_error2) * isw)
                critic_loss = 0.5 * q1_loss + 0.5 * q2_loss

                z_a = vis_feat  # [B, N]
                z_out = self.encoder_target(pos)
                logits = tf.matmul(z_a, tf.matmul(self.curl_w, tf.transpose(z_out, [1, 0])))
                logits -= tf.reduce_max(logits, axis=-1, keepdims=True)
                curl_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.range(self.batch_size), logits))
            critic_grads = tape.gradient(critic_loss, self.critic_tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_tv)
            )
            curl_grads = tape.gradient(curl_loss, [self.curl_w] + self.encoder.trainable_variables)
            self.optimizer_curl.apply_gradients(
                zip(curl_grads, [self.curl_w] + self.encoder.trainable_variables)
            )

            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu, log_std = self.actor_net.value_net(feat)
                    log_std = clip_nn_log_std(log_std, self.log_std_min, self.log_std_max)
                    pi, log_pi = squash_rsample(mu, log_std)
                    entropy = gaussian_entropy(log_std)
                else:
                    logits = self.actor_net.value_net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    gumbel_noise = tf.cast(self.gumbel_dist.sample([batch_size, self.a_dim]), dtype=tf.float32)
                    _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_dim)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    pi = _pi_diff + _pi
                    log_pi = tf.reduce_sum(tf.multiply(logp_all, pi), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                q_s_pi = self.critic_net.get_min(feat, pi)
                actor_loss = -tf.reduce_mean(q_s_pi - self.alpha * log_pi)
            actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.trainable_variables)
            )

            if self.auto_adaption:
                with tf.GradientTape() as tape:
                    if self.is_continuous:
                        mu, log_std = self.actor_net.value_net(feat)
                        log_std = clip_nn_log_std(log_std, self.log_std_min, self.log_std_max)
                        norm_dist = tfp.distributions.Normal(loc=mu, scale=tf.exp(log_std))
                        log_pi = tf.reduce_sum(norm_dist.log_prob(norm_dist.sample()), axis=-1)
                    else:
                        logits = self.actor_net.value_net(feat)
                        cate_dist = tfp.distributions.Categorical(logits=tf.nn.log_softmax(logits))
                        log_pi = cate_dist.log_prob(cate_dist.sample())
                    alpha_loss = -tf.reduce_mean(self.alpha * tf.stop_gradient(log_pi + self.target_entropy))
                alpha_grad = tape.gradient(alpha_loss, self.log_alpha)
                self.optimizer_alpha.apply_gradients(
                    [(alpha_grad, self.log_alpha)]
                )
            self.global_step.assign_add(1)
            summaries = dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/q1_loss', q1_loss],
                ['LOSS/q2_loss', q2_loss],
                ['LOSS/critic_loss', critic_loss],
                ['LOSS/curl_loss', curl_loss],
                ['Statistics/log_alpha', self.log_alpha],
                ['Statistics/alpha', self.alpha],
                ['Statistics/entropy', entropy],
                ['Statistics/q_min', tf.reduce_min(tf.minimum(q1, q2))],
                ['Statistics/q_mean', tf.reduce_mean(tf.minimum(q1, q2))],
                ['Statistics/q_max', tf.reduce_max(tf.maximum(q1, q2))]
            ])
            if self.auto_adaption:
                summaries.update({
                    'LOSS/alpha_loss': alpha_loss
                })
            return (td_error1 + td_error2) / 2., summaries
