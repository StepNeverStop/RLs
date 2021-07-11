#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from torch import distributions as td
from torch.nn import (Sequential,
                      Linear,
                      LayerNorm)
from skimage.util.shape import view_as_windows

from rls.utils.torch_utils import (squash_rsample,
                                   gaussian_entropy,
                                   q_target_func,
                                   sync_params_pairs)
from rls.algos.base.off_policy import Off_Policy
from rls.utils.sundry_utils import LinearAnnealing
from rls.common.specs import BatchExperiences
from rls.nn.visual_nets import Vis_REGISTER
from rls.nn.models import (CriticQvalueOne,
                           ActorCts,
                           ActorDct)
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


class VisualEncoder(t.nn.Module):

    def __init__(self, img_dim, fc_dim):
        super().__init__()
        net = Vis_REGISTER['nature'](visual_dim=img_dim)
        self.net = Sequential(
            net,
            Linear(net.output_dim, fc_dim),
            LayerNorm(fc_dim)
        )

    def forward(self, vis):
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
                 network_settings={
                     'actor_continuous': {
                         'share': [128, 128],
                         'mu': [64],
                         'log_std': [64],
                         'soft_clip': False,
                         'log_std_bound': [-20, 2]
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
        self.concat_vector_dim = self.obs_spec.total_vector_dim
        self.ployak = ployak
        self.discrete_tau = discrete_tau
        self.auto_adaption = auto_adaption
        self.annealing = annealing
        self.img_size = img_size
        self.img_dim = [img_size, img_size, self.obs_spec.visual_dims[0][-1]]
        self.vis_feat_size = network_settings['encoder']

        if self.auto_adaption:
            self.log_alpha = t.tensor(0., requires_grad=True)
        else:
            self.log_alpha = t.tensor(alpha).log()
            if self.annealing:
                self.alpha_annealing = LinearAnnealing(alpha, last_alpha, 1.0e6)

        self.critic = CriticQvalueOne(vector_dim=self.concat_vector_dim + self.vis_feat_size,
                                      action_dim=self.a_dim,
                                      network_settings=network_settings['q'])

        self.critic2 = CriticQvalueOne(vector_dim=self.concat_vector_dim + self.vis_feat_size,
                                       action_dim=self.a_dim,
                                       network_settings=network_settings['q'])
        self.critic_target = deepcopy(self.critic)
        self.critic_target.eval()
        self.critic2_target = deepcopy(self.critic2)
        self.critic2_target.eval()

        if self.is_continuous:
            self.actor = ActorCts(vector_dim=self.concat_vector_dim + self.vis_feat_size,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_continuous'])
        else:
            self.actor = ActorDct(vector_dim=self.concat_vector_dim + self.vis_feat_size,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_discrete'])
            self.gumbel_dist = td.gumbel.Gumbel(0, 1)

        # entropy = -log(1/|A|) = log |A|
        self.target_entropy = 0.98 * (-self.a_dim if self.is_continuous else np.log(self.a_dim))

        self.encoder = VisualEncoder(self.img_dim, self.vis_feat_size)
        self.encoder_target = VisualEncoder(self.img_dim, self.vis_feat_size)

        self.curl_w = t.tensor(t.randn(self.vis_feat_size, self.vis_feat_size), requires_grad=True)

        self._pairs = [(self.critic_target, self.critic),
                       (self.critic2_target, self.critic2),
                       (self.encoder_target, self.encoder)]
        sync_params_pairs(self._pairs)

        self.actor_oplr = OPLR(self.actor, actor_lr)
        self.critic_oplr = OPLR([self.critic, self.critic2, self.encoder], critic_lr)
        self.alpha_oplr = OPLR(self.log_alpha, alpha_lr)
        self.curl_oplr = OPLR([self.w, self.encoder], curl_lr)

        self._worker_modules.update(actor=self.actor,
                                    encoder=self.encoder)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(critic=self.critic,
                                     curl_w=self.curl_w,
                                     actor_oplr=self.actor_oplr,
                                     critic_oplr=self.critic_oplr,
                                     alpha_oplr=self.alpha_oplr,
                                     curl_oplr=self.curl_oplr)
        self.initialize_data_buffer()

    @iTensor_oNumpy
    def __call__(self, obs, evaluation=False):
        visual = center_crop_image(obs.first_visual()[:, 0], self.img_size)
        feat = t.cat([self.encoder(visual), obs.flatten_vector()], -1)
        if self.is_continuous:
            mu, log_std = self.actor(feat)
            pi, _ = squash_rsample(mu, log_std)
            mu.tanh_()  # squash mu
        else:
            logits = self.actor(feat)
            mu = logits.argmax(1)
            cate_dist = td.categorical.Categorical(logits=logits)
            pi = cate_dist.sample()
        return mu if evaluation else pi

    def _process_before_train(self, data: BatchExperiences):
        visual = np.transpose(data.obs.first_visual()[:, 0].numpy(), (0, 3, 1, 2))
        visual_ = np.transpose(data.obs_.first_visual()[:, 0].numpy(), (0, 3, 1, 2))
        pos = np.transpose(random_crop(visual, self.img_size), (0, 2, 3, 1))
        visual = np.transpose(random_crop(visual, self.img_size), (0, 2, 3, 1))
        visual_ = np.transpose(random_crop(visual_, self.img_size), (0, 2, 3, 1))
        return self.data_convert([visual, visual_, pos])

    def _target_params_update(self):
        sync_params_pairs(self._pairs, self.ployak)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([
                    ['LEARNING_RATE/actor_lr', self.actor_oplr.lr],
                    ['LEARNING_RATE/critic_lr', self.critic_oplr.lr],
                    ['LEARNING_RATE/alpha_lr', self.alpha_oplr.lr]
                ])
            })

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _train(self, BATCH: BatchExperiences, isw, cell_states):
        visual, visual_, pos = self._process_before_train(BATCH)
        td_error, summaries = self.train(BATCH, isw, cell_states, visual, visual_, pos)
        if self.annealing and not self.auto_adaption:
            self.log_alpha.copy_(self.alpha_annealing(self.global_step).log())
        return td_error, summaries

    @iTensor_oNumpy
    def train(self, BATCH, isw, cell_states, visual, visual_, pos):
        vis_feat = self.encoder(visual)
        vis_feat_ = self.encoder(visual_)
        target_vis_feat_ = self.encoder_target(visual_)
        feat = t.cat([vis_feat, BATCH.obs.flatten_vector()], -1)
        feat_ = t.cat([vis_feat_, BATCH.obs_.flatten_vector()], -1)
        target_feat_ = t.cat([target_vis_feat_, BATCH.obs_.flatten_vector()], -1)
        if self.is_continuous:
            target_mu, target_log_std = self.actor(feat_)
            target_pi, target_log_pi = squash_rsample(target_mu, target_log_std)
        else:
            target_logits = self.actor(feat_)
            target_cate_dist = td.categorical.Categorical(logits=target_logits)
            target_pi = target_cate_dist.sample()
            target_log_pi = target_cate_dist.log_prob(target_pi)
            target_pi = t.nn.functional.one_hot(target_pi, self.a_dim).float()
        q1 = self.critic(feat, BATCH.action)
        q2 = self.critic2(feat, BATCH.action)
        q1_target = self.critic_target(feat_, target_pi)
        q2_target = self.critic2_target(feat_, target_pi)
        q_target = t.minimum(q1_target, q2_target)
        dc_r = q_target_func(BATCH.reward,
                             self.gamma,
                             BATCH.done,
                             (q_target - self.alpha * target_log_pi))
        td_error1 = q1 - dc_r
        td_error2 = q2 - dc_r
        q1_loss = (td_error1.square() * isw).mean()
        q2_loss = (td_error2.square() * isw).mean()
        critic_loss = 0.5 * q1_loss + 0.5 * q2_loss

        z_a = vis_feat  # [B, N]
        z_out = self.encoder_target(pos)
        logits = z_a @ (self.curl_w @ z_out.T)
        logits -= logits.max(-1, keepdim=True)[0]
        curl_loss = t.nn.functional.cross_entropy(logits, t.arange(self.batch_size))

        self.critic_oplr.step(critic_loss)
        self.curl_oplr.step(curl_loss)

        if self.is_continuous:
            mu, log_std = self.actor(feat)
            pi, log_pi = squash_rsample(mu, log_std)
            entropy = gaussian_entropy(log_std)
        else:
            logits = self.actor(feat)
            logp_all = logits.log_softmax(-1)
            gumbel_noise = self.gumbel_dist.sample(BATCH.action.shape)
            _pi = ((logp_all + gumbel_noise) / self.discrete_tau).softmax(-1)
            _pi_true_one_hot = t.nn.functional.one_hot(_pi.argmax(-1), self.a_dim).float()
            _pi_diff = (_pi_true_one_hot - _pi).detach()
            pi = _pi_diff + _pi
            log_pi = (logp_all * pi).sum(1, keepdim=True)
            entropy = -(logp_all.exp() * logp_all).sum(1, keepdim=True).mean()
        q_s_pi = t.minimum(self.critic(feat, pi), self.critic2(feat, pi))
        actor_loss = -(q_s_pi - self.alpha * log_pi).mean()
        self.actor_oplr.step(actor_loss)

        if self.auto_adaption:
            if self.is_continuous:
                mu, log_std = self.actor(feat)
                norm_dist = td.normal.Normal(loc=mu, scale=log_std.exp())
                log_pi = norm_dist.log_prob(norm_dist.sample()).sum(-1, keepdim=True)  # [B, 1]
            else:
                logits = self.actor(feat)
                norm_dist = td.categorical.Categorical(logits=logits)
                log_pi = norm_dist.log_prob(cate_dist.sample())
            alpha_loss = -(self.alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_oplr.step(alpha_loss)

        self.global_step.add_(1)
        summaries = dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/q1_loss', q1_loss],
            ['LOSS/q2_loss', q2_loss],
            ['LOSS/critic_loss', critic_loss],
            ['LOSS/curl_loss', curl_loss],
            ['Statistics/log_alpha', self.log_alpha],
            ['Statistics/alpha', self.alpha],
            ['Statistics/entropy', entropy],
            ['Statistics/q_min', t.minimum(q1, q2).min()],
            ['Statistics/q_mean', t.minimum(q1, q2).mean()],
            ['Statistics/q_max', t.maximum(q1, q2).max()]
        ])
        if self.auto_adaption:
            summaries.update({
                'LOSS/alpha_loss': alpha_loss
            })
        return (td_error1 + td_error2) / 2., summaries
