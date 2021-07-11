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
from rls.nn.models import (ActorMuLogstd,
                           ActorDct,
                           CriticValue)
from rls.nn.utils import OPLR
from rls.utils.sundry_utils import to_numpy


@dataclass(eq=False)
class TRPO_Store_BatchExperiences_CTS(BatchExperiences):
    value: np.ndarray
    log_prob: np.ndarray
    mu: np.ndarray
    log_std: np.ndarray


@dataclass(eq=False)
class TRPO_Train_BatchExperiences_CTS(Data):
    obs: ModelObservations
    action: np.ndarray
    log_prob: np.ndarray
    discounted_reward: np.ndarray
    gae_adv: np.ndarray
    mu: np.ndarray
    log_std: np.ndarray


@dataclass(eq=False)
class TRPO_Store_BatchExperiences_DCT(BatchExperiences):
    value: np.ndarray
    log_prob: np.ndarray
    mu: np.ndarray
    log_std: np.ndarray


@dataclass(eq=False)
class TRPO_Train_BatchExperiences_DCT(Data):
    obs: ModelObservations
    action: np.ndarray
    log_prob: np.ndarray
    discounted_reward: np.ndarray
    gae_adv: np.ndarray
    logp_all: np.ndarray


'''
Stole this from OpenAI SpinningUp. https://github.com/openai/spinningup/blob/master/spinup/algos/trpo/trpo.py
'''


def flat_concat(xs):
    return t.cat([x.flatten() for x in xs], 0)


def assign_params_from_flat(x, params):
    def flat_size(p): return int(np.prod(p.shape.as_list()))  # the 'int' is important for scalars
    splits = x.split([flat_size(p) for p in params])
    new_params = [p_new.view_as(p) for p, p_new in zip(params, splits)]
    [p.data.copy_(p_new) for p, p_new in zip(params, new_params)]


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
            self.actor = ActorMuLogstd(self.rep_net.h_dim,
                                       output_shape=self.a_dim,
                                       network_settings=network_settings['actor_continuous'])
        else:
            self.actor = ActorDct(self.rep_net.h_dim,
                                  output_shape=self.a_dim,
                                  network_settings=network_settings['actor_discrete'])
        self.critic = CriticValue(self.rep_net.h_dim,
                                  network_settings=network_settings['critic'])

        self.critic_oplr = OPLR([self.critic, self.rep_net], critic_lr)

        if self.is_continuous:
            self.initialize_data_buffer(store_data_type=TRPO_Store_BatchExperiences_CTS,
                                        sample_data_type=TRPO_Train_BatchExperiences_CTS)
        else:
            self.initialize_data_buffer(store_data_type=TRPO_Store_BatchExperiences_DCT,
                                        sample_data_type=TRPO_Train_BatchExperiences_DCT)

        self._worker_modules.update(rep_net=self.rep_net,
                                    actor=self.actor)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(critic=self.critic,
                                     critic_oplr=self.critic_oplr)

    def __call__(self, obs, evaluation=False):
        a, self.next_cell_state = self._get_action(obs, self.cell_state)
        return a

    def _get_action(self, obs, cell_state):
        feat, cell_state = self.rep_net(obs.tensor, cell_state=cell_state)
        self._value = to_numpy(self.critic(feat))
        if self.is_continuous:
            mu, log_std = output
            self._mu = to_numpy(mu)
            self._log_std = to_numpy(log_std)
            sample_op, _ = gaussian_clip_rsample(mu, log_std)
            log_prob = gaussian_likelihood_sum(sample_op, mu, log_std)
        else:
            logits = output
            logp_all = logits.log_softmax(-1)
            self._logp_all = to_numpy(logp_all)
            norm_dist = td.categorical.Categorical(logits=logp_all)
            sample_op = norm_dist.sample()
            log_prob = norm_dist.log_prob(sample_op)
        self._log_prob = to_numpy(log_prob) + 1e-10
        return to_numpy(sample_op), cell_state

    def store_data(self, exps: BatchExperiences):
        # self._running_average()

        if self.is_continuous:
            self.data.add(TRPO_Store_BatchExperiences_CTS(*exps.astuple(), self._value, self._log_prob, self._mu, self._log_std))
        else:
            self.data.add(TRPO_Store_BatchExperiences_DCT(*exps.astuple(), self._value, self._log_prob, self._logp_all))
        if self.use_rnn:
            self.data.add_cell_state(tuple(cs.numpy() for cs in self.cell_state))
        self.cell_state = self.next_cell_state

    def _get_value(self, obs, cell_state):
        feat, cell_state = self.rep_net(obs.tensor, cell_state=cell_state)
        value = self.critic(feat)
        return value, cell_state

    def calculate_statistics(self):
        init_value, self.cell_state = self._get_value(self.data.last_data().obs_, cell_state=self.cell_state)
        init_value = init_value.numpy()
        self.data.cal_dc_r(self.gamma, init_value)
        self.data.cal_td_error(self.gamma, init_value)
        self.data.cal_gae_adv(self.lambda_, self.gamma)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        def _train(data, cell_state):
            actor_loss, entropy, gradients = self.train_actor(data, cell_state)

            x = self.cg(self.Hx, gradients.numpy(), data, cell_state)
            x = t.tensor(x)
            alpha = np.sqrt(2 * self.delta / (np.dot(x, self.Hx(x, data, cell_state)) + 1e-8))
            for i in range(self.backtrack_iters):
                assign_params_from_flat(alpha * x * (self.backtrack_coeff ** i), self.actor)

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
                ['LEARNING_RATE/critic_lr', self.critic_oplr.lr]
            ])
        })

    def train_actor(self, BATCH, cell_state):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_state['obs'])
        output = self.actor(feat)
        if self.is_continuous:
            mu, log_std = output
            new_log_prob = gaussian_likelihood_sum(BATCH.action, mu, log_std)
            entropy = gaussian_entropy(log_std)
        else:
            logits = output
            logp_all = logits.log_softmax(-1)
            new_log_prob = (BATCH.action * logp_all).sum(1, keepdim=True)
            entropy = -(logp_all.exp() * logp_all).sum(1, keepdim=True).mean()
        ratio = (new_log_prob - BATCH.log_prob).exp()
        actor_loss = -(ratio * BATCH.gae_adv).mean()
        actor_grads = tape.gradient(actor_loss, self.actor)    # TODO
        gradients = flat_concat(actor_grads)
        self.global_step.add_(1)
        return actor_loss, entropy, gradients

    def Hx(self, x, BATCH, cell_state):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_state['obs'])
        output = self.actor(feat)
        if self.is_continuous:
            mu, log_std = output
            var0, var1 = (2 * log_std).exp(), (2 * BATCH.log_std).exp()
            pre_sum = 0.5 * (((BATCH.mu - mu)**2 + var0) / (var1 + 1e-8) - 1) + BATCH.log_std - log_std
            all_kls = pre_sum.sum(1)
        else:
            logits = output
            logp_all = logits.log_softmax(-1)
            all_kls = (BATCH.logp_all.exp() * (BATCH.logp_all - logp_all)).sum(1)
        kl = all_kls.mean()
        g = flat_concat(tape.gradient(kl, self.actor))
        _g = (g * x).sum()
        hvp = flat_concat(tape.gradient(_g, self.actor))
        if self.damping_coeff > 0:
            hvp += self.damping_coeff * x
        return hvp

    def train_critic(self, BATCH, cell_state):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_state['obs'])
        value = self.critic(feat)
        td_error = BATCH.discounted_reward - value
        value_loss = td_error.square().mean()
        self.critic_oplr.step(value_loss)
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
            z = Ax(t.tensor(p), BATCH, cell_state)
            alpha = r_dot_old / (np.dot(p, z) + 1e-8)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x
