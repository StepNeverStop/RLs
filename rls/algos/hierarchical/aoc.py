#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from torch import distributions as td
from dataclasses import dataclass

from rls.utils.torch_utils import (gaussian_clip_rsample,
                                   gaussian_likelihood_sum,
                                   gaussian_entropy)
from rls.algos.base.on_policy import On_Policy
from rls.utils.specs import (ModelObservations,
                             Data,
                             BatchExperiences)
from rls.nn.models import AocShare
from rls.nn.utils import OPLR
from rls.utils.sundry_utils import to_numpy


@dataclass(eq=False)
class AOC_Store_BatchExperiences(BatchExperiences):
    value: np.ndarray
    log_prob: np.ndarray
    beta_advantage: np.ndarray
    last_options: np.ndarray
    options: np.ndarray


@dataclass(eq=False)
class AOC_Train_BatchExperiences(Data):
    obs: ModelObservations
    action: np.ndarray
    value: np.ndarray
    log_prob: np.ndarray
    discounted_reward: np.ndarray
    gae_adv: np.ndarray
    beta_advantage: np.ndarray
    last_options: np.ndarray
    options: np.ndarray


class AOC(On_Policy):
    '''
    Asynchronous Advantage Option-Critic with Deliberation Cost, A2OC
    When Waiting is not an Option : Learning Options with a Deliberation Cost, A2OC, http://arxiv.org/abs/1709.04571
    '''

    def __init__(self,
                 envspec,

                 options_num=4,
                 dc=0.01,
                 terminal_mask=False,
                 eps=0.1,
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
                     'termination': [32, 32]
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
        self.kl_coef = t.tensor(kl_coef).float()

        self.kl_cutoff = kl_target * kl_target_cutoff
        self.kl_stop = kl_target * kl_target_earlystop
        self.kl_low = kl_target * kl_beta[0]
        self.kl_high = kl_target * kl_beta[-1]

        self.options_num = options_num
        self.dc = dc
        self.terminal_mask = terminal_mask
        self.eps = eps

        self.net = AocShare(self.rep_net.h_dim,
                            action_dim=self.a_dim,
                            options_num=self.options_num,
                            network_settings=network_settings,
                            is_continuous=self.is_continuous)
        if self.is_continuous:
            self.log_std = -0.5 * t.ones((self.options_num, self.a_dim), requires_grad=True)   # [P, A]
        self.oplr = OPLR([self.net, self.rep_net, self.log_std], lr)

        self._worker_modules.update(rep_net=self.rep_net,
                                    model=self.net)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(oplr=self.oplr)

        self.initialize_data_buffer(store_data_type=AOC_Store_BatchExperiences,
                                    sample_data_type=AOC_Train_BatchExperiences)

    def reset(self):
        super().reset()
        self._done_mask = np.full(self.n_copys, True)

    def partial_reset(self, done):
        super().partial_reset(done)
        self._done_mask = done

    def _generate_random_options(self):
        return t.tensor(np.random.randint(0, self.options_num, self.n_copys)).int()

    def __call__(self, obs, evaluation=False):
        if not hasattr(self, 'options'):
            self.options = self._generate_random_options()
        self.last_options = self.options
        if not hasattr(self, 'oc_mask'):
            self.oc_mask = t.tensor(np.zeros(self.n_copys)).int()

        a = self._get_action(obs, self.cell_state, self.options)
        return a

    def _get_action(self, obs, cell_state, options):
        feat, cell_state = self.rep_net(obs.tensor, cell_state=cell_state)  # [B, P], [B, P, A], [B, P], [B, P]
        (q, pi, beta) = self.net(feat)
        options_onehot = t.nn.functional.one_hot(options, self.options_num).float()    # [B, P]
        options_onehot_expanded = options_onehot.unsqueeze(-1)  # [B, P, 1]
        pi = (pi * options_onehot_expanded).sum(1)  # [B, A]
        if self.is_continuous:
            mu = pi
            log_std = self.log_std[options]
            sample_op, _ = gaussian_clip_rsample(mu, log_std)
            log_prob = gaussian_likelihood_sum(sample_op, mu, log_std)
        else:
            logits = pi
            norm_dist = td.categorical.Categorical(logits=logits)
            sample_op = norm_dist.sample()
            log_prob = norm_dist.log_prob(sample_op)
        value = q_o = (q * options_onehot).sum(-1, keepdim=True)  # [B, 1]
        beta_adv = q_o - ((1 - self.eps) * q.max(-1, keepdim=True)[0] + self.eps * q.mean(-1, keepdim=True))   # [B, 1]
        max_options = q.argmax(-1)  # [B, P] => [B, ]
        beta_probs = (beta * options_onehot).sum(1)   # [B, P] => [B,]
        beta_dist = td.bernoulli.Bernoulli(probs=beta_probs)
        new_options = t.where(beta_dist.sample() < 1, options, max_options)    # <1 则不改变op， =1 则改变op

        new_options = t.where(self._done_mask, max_options, new_options)
        self._done_mask = np.full(self.n_copys, False)
        self._value = to_numpy(value)
        self._log_prob = to_numpy(log_prob) + 1e-10
        self._beta_adv = to_numpy(beta_adv) + self.dc
        self.oc_mask = to_numpy(new_options == self.options)  # equal means no change
        self.options = to_numpy(new_options)
        self.next_cell_state = cell_state
        return to_numpy(sample_op)

    def store_data(self, exps: BatchExperiences):
        # self._running_average()
        exps.reward = exps.reward - ((1 - self.oc_mask) * self.dc).unsqueeze(-1)
        self.data.add(AOC_Store_BatchExperiences(*exps.astuple(), self._value, self._log_prob, self._beta_adv,
                                                 self.last_options, self.options))
        if self.use_rnn:
            self.data.add_cell_state(tuple(cs.numpy() for cs in self.cell_state))
        self.cell_state = self.next_cell_state
        self.oc_mask = np.zeros_like(self.oc_mask)

    def _get_value(self, obs, options, cell_state):
        feat, cell_state = self.rep_net(obs.tensor, cell_state=cell_state)
        (q, _, _) = self.net(feat)
        options_onehot = t.nn.functional.one_hot(options, self.options_num).float()    # [B, P]
        value = q_o = (q * options_onehot).sum(-1, keepdim=True)  # [B, 1]
        return to_numpy(value), cell_state

    def calculate_statistics(self):
        last_data = self.data.last_data()
        init_value, self.cell_state = self._get_value(last_data.obs_, last_data.options, cell_state=self.cell_state)
        self.data.cal_dc_r(self.gamma, init_value)
        self.data.cal_td_error(self.gamma, init_value)
        self.data.cal_gae_adv(self.lambda_, self.gamma)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        def _train(data, cell_state):
            early_step = 0
            for i in range(self.epoch):
                loss, pi_loss, q_loss, beta_loss, entropy, kl = self.train(data.tensor, cell_state, self.kl_coef)
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
                ['LOSS/loss', beta_loss],
                ['Statistics/kl', kl],
                ['Statistics/entropy', entropy],
                ['Statistics/kl_coef', self.kl_coef],
                ['Statistics/early_step', early_step],
            ])
            return summaries

        summary_dict = dict([['LEARNING_RATE/lr', self.oplr.lr]])

        self._learn(function_dict={
            'calculate_statistics': self.calculate_statistics,
            'train_function': _train,
            'summary_dict': summary_dict
        })

    def train(self, BATCH, cell_state, kl_coef):
        last_options = BATCH.last_options  # [B,]
        options = BATCH.options
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_state['obs'])  # [B, P], [B, P, A], [B, P], [B, P]
        (q, pi, beta) = self.net(feat)
        options_onehot = t.nn.functional.one_hot(options, self.options_num).float()    # [B, P]
        options_onehot_expanded = options_onehot.unsqueeze(-1)  # [B, P, 1]
        last_options_onehot = t.nn.functional.one_hot(last_options, self.options_num).float()    # [B,] => [B, P]

        pi = (pi * options_onehot_expanded).sum(1)  # [B, P, A] => [B, A]
        value = (q * options_onehot).sum(1, keepdim=True)    # [B, 1]

        if self.is_continuous:
            mu = pi  # [B, A]
            log_std = self.log_std[options]
            new_log_prob = gaussian_likelihood_sum(BATCH.action, mu, log_std)
            entropy = gaussian_entropy(log_std)
        else:
            logits = pi  # [B, A]
            logp_all = logits.log_softmax(-1)
            new_log_prob = (BATCH.action * logp_all).sum(1, keepdim=True)
            entropy = -(logp_all.exp() * logp_all).sum(1, keepdim=True).mean()
        ratio = (new_log_prob - BATCH.log_prob).exp()

        if self.kl_reverse:
            kl = (new_log_prob - BATCH.log_prob).mean()
        else:
            kl = (BATCH.log_prob - new_log_prob).mean()    # a sample estimate for KL-divergence, easy to compute
        surrogate = ratio * BATCH.gae_adv

        value_clip = BATCH.value + (value - BATCH.value).clamp(-self.value_epsilon, self.value_epsilon)
        td_error = BATCH.discounted_reward - value
        td_error_clip = BATCH.discounted_reward - value_clip
        td_square = t.maximum(td_error.square(), td_error_clip.square())

        pi_loss = -t.minimum(
            surrogate,
            ratio.clamp(1.0 - self.epsilon, 1.0 + self.epsilon) * BATCH.gae_adv
        ).mean()
        kl_loss = kl_coef * kl
        extra_loss = 1000.0 * t.maximum(t.zeros_like(kl), kl - self.kl_cutoff).square()
        pi_loss = pi_loss + kl_loss + extra_loss
        q_loss = 0.5 * td_square.mean()

        beta_s = (beta * last_options_onehot).sum(-1, keepdim=True)   # [B, 1]
        beta_loss = (beta_s * BATCH.beta_advantage).mean()
        if self.terminal_mask:
            beta_loss *= (1 - done)

        loss = pi_loss + 1.0 * q_loss + beta_loss - self.pi_beta * entropy
        self.oplr.step(loss)
        self.global_step.add_(1)
        return loss, pi_loss, q_loss, beta_loss, entropy, kl
