#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from torch import distributions as td
from dataclasses import dataclass

from rls.algorithms.base.off_policy import Off_Policy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import (q_target_func,
                                   sync_params_pairs)
from rls.common.specs import BatchExperiences
from rls.nn.models import (OcIntraOption,
                           CriticQvalueAll)
from rls.nn.utils import OPLR
from rls.utils.converter import to_numpy
from rls.common.decorator import iTensor_oNumpy


@dataclass(eq=False)
class OC_BatchExperiences(BatchExperiences):
    last_options: np.ndarray
    options: np.ndarray


class OC(Off_Policy):
    '''
    The Option-Critic Architecture. http://arxiv.org/abs/1609.05140
    '''

    def __init__(self,
                 envspec,

                 q_lr=5.0e-3,
                 intra_option_lr=5.0e-4,
                 termination_lr=5.0e-4,
                 use_eps_greedy=False,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
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
                     'termination': [32, 32]
                 },
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval
        self.options_num = options_num
        self.termination_regularizer = termination_regularizer
        self.ent_coff = ent_coff
        self.use_baseline = use_baseline
        self.terminal_mask = terminal_mask
        self.double_q = double_q
        self.boltzmann_temperature = boltzmann_temperature
        self.use_eps_greedy = use_eps_greedy

        self.q_net = CriticQvalueAll(self.rep_net.h_dim,
                                     output_shape=self.options_num,
                                     network_settings=network_settings['q']).to(self.device)
        self.q_target_net = deepcopy(self.q_net)
        self.q_target_net.eval()

        self._target_rep_net = deepcopy(self.rep_net)
        self._target_rep_net.eval()

        self.intra_option_net = OcIntraOption(vector_dim=self.rep_net.h_dim,
                                              output_shape=self.a_dim,
                                              options_num=self.options_num,
                                              network_settings=network_settings['intra_option']).to(self.device)
        self.termination_net = CriticQvalueAll(vector_dim=self.rep_net.h_dim,
                                               output_shape=self.options_num,
                                               network_settings=network_settings['termination'],
                                               out_act='sigmoid').to(self.device)

        if self.is_continuous:
            self.log_std = -0.5 * t.ones((self.options_num, self.a_dim), requires_grad=True)   # [P, A]

        self._pairs = [(self.q_target_net, self.q_net),
                       (self._target_rep_net, self.rep_net)]
        sync_params_pairs(self._pairs)

        self.q_oplr = OPLR([self.rep_net, self.q_net], q_lr, clipvalue=5.)
        self.intra_option_oplr = OPLR([self.intra_option_net, self.log_std], intra_option_lr, clipvalue=5.)
        self.termination_oplr = OPLR(self.termination_net, termination_lr, clipvalue=5.)

        self._worker_modules.update(rep_net=self.rep_net,
                                    q_net=self.q_net,
                                    intra_option_net=self.intra_option_net,
                                    termination_net=self.termination_net)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(q_op=self.q_op,
                                     intra_option_oplr=self.intra_option_oplr,
                                     termination_oplr=self.termination_oplr)
        self.initialize_data_buffer()
        self.options = self._generate_random_options()

    def _generate_random_options(self):
        return t.tensor(np.random.randint(0, self.options_num, self.n_copys)).int()

    @iTensor_oNumpy
    def __call__(self, obs, evaluation=False):
        self.last_options = self.options

        feat, self.cell_state = self.rep_net(obs, cell_state=self.cell_state)
        q = self.q_net(feat)  # [B, P]
        pi = self.intra_option_net(feat)  # [B, P, A]
        beta = self.termination_net(feat)  # [B, P]
        options_onehot = t.nn.functional.one_hot(self.options, self.options_num).float()    # [B, P]
        options_onehot_expanded = options_onehot.unsqueeze(-1)  # [B, P, 1]
        pi = (pi * options_onehot_expanded).sum(1)  # [B, A]
        if self.is_continuous:
            mu = pi.tanh()
            log_std = self.log_std[self.options]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            a = dist.sample().clamp(-1, 1)
        else:
            pi = pi / self.boltzmann_temperature
            dist = td.Categorical(logits=pi)  # [B, ]
            a = dist.sample()
        max_options = q.argmax(-1).int()  # [B, P] => [B, ]
        if self.use_eps_greedy:
            new_options = max_options
        else:
            beta_probs = (beta * options_onehot).sum(1)   # [B, P] => [B,]
            beta_dist = td.Bernoulli(probs=beta_probs)
            new_options = t.where(beta_dist.sample() < 1, self.options, max_options)
        self.options = to_numpy(new_options)

        if self.use_eps_greedy:
            if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):   # epsilon greedy
                self.options = self._generate_random_options()
        return a

    def _target_params_update(self):
        if self.global_step % self.assign_interval == 0:
            sync_params_pairs(self._pairs)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([
                    ['LEARNING_RATE/q_lr', self.q_oplr.lr],
                    ['LEARNING_RATE/intra_option_lr', self.intra_option_oplr.lr],
                    ['LEARNING_RATE/termination_lr', self.termination_oplr.lr],
                    ['Statistics/option', self.options[0]]
                ])
            })

    @iTensor_oNumpy
    def _train(self, BATCH, isw, cell_states):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
        feat_, _ = self._target_rep_net(BATCH.obs_, cell_state=cell_states['obs_'])

        last_options = BATCH.last_options
        options = BATCH.options
        q = self.q_net(feat)  # [B, P]
        q_next = self.q_target_net(feat_)   # [B, P], [B, P, A], [B, P]
        beta_next = self.termination_net(feat_)  # [B, P]
        options_onehot = t.nn.functional.one_hot(options, self.options_num).float()    # [B,] => [B, P]

        q_s = qu_eval = (q * options_onehot).sum(-1, keepdim=True)  # [B, 1]
        beta_s_ = (beta_next * options_onehot).sum(-1, keepdim=True)  # [B, 1]
        q_s_ = (q_next * options_onehot).sum(-1, keepdim=True)   # [B, 1]
        # https://github.com/jeanharb/option_critic/blob/5d6c81a650a8f452bc8ad3250f1f211d317fde8c/neural_net.py#L94
        if self.double_q:
            feat__, _ = self.rep_net(BATCH.obs_, cell_state=cell_states['obs_'])
            q_ = self.q_net(feat__)  # [B, P], [B, P, A], [B, P]
            max_a_idx = t.nn.functional.one_hot(q_.argmax(-1), self.options_num).float()  # [B, P] => [B, ] => [B, P]
            q_s_max = (q_next * max_a_idx).sum(-1, keepdim=True)   # [B, 1]
        else:
            q_s_max = q_next.max(-1, keepdim=True)[0]   # [B, 1]
        u_target = (1 - beta_s_) * q_s_ + beta_s_ * q_s_max   # [B, 1]
        qu_target = q_target_func(BATCH.reward,
                                  self.gamma,
                                  BATCH.done,
                                  u_target)
        td_error = qu_target - qu_eval     # gradient : q
        q_loss = (td_error.square() * isw).mean()        # [B, 1] => 1
        self.q_oplr.step(q_loss)

        feat = feat.detach()
        pi = self.intra_option_net(feat)  # [B, P, A]
        beta = self.termination_net(feat)   # [B, P]

        # https://github.com/jeanharb/option_critic/blob/5d6c81a650a8f452bc8ad3250f1f211d317fde8c/neural_net.py#L130
        if self.use_baseline:
            adv = (qu_target - qu_eval).detach()
        else:
            adv = qu_target.detach()
        options_onehot_expanded = options_onehot.unsqueeze(-1)   # [B, P] => [B, P, 1]
        pi = (pi * options_onehot_expanded).sum(1)  # [B, P, A] => [B, A]
        if self.is_continuous:
            mu = pi.tanh()
            log_std = self.log_std[options]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            log_p = dist.log_prob(BATCH.action).unsqueeze(-1)
            entropy = dist.entropy().mean()
        else:
            pi = pi / self.boltzmann_temperature
            log_pi = pi.log_softmax(-1)  # [B, A]
            entropy = -(log_pi.exp() * log_pi).sum(1, keepdim=True)    # [B, 1]
            log_p = (BATCH.action * log_pi).sum(-1, keepdim=True)   # [B, 1]
        pi_loss = -(log_p * adv + self.ent_coff * entropy).mean()              # [B, 1] * [B, 1] => [B, 1] => 1

        last_options_onehot = t.nn.functional.one_hot(last_options, self.options_num).float()    # [B,] => [B, P]
        beta_s = (beta * last_options_onehot).sum(-1, keepdim=True)   # [B, 1]
        if self.use_eps_greedy:
            v_s = q.max(-1, keepdim=True)[0] - self.termination_regularizer   # [B, 1]
        else:
            v_s = (1 - beta_s) * q_s + beta_s * q.max(-1, keepdim=True)[0]    # [B, 1]
            # v_s = q.mean(-1, keepdim=True)   # [B, 1]
        beta_loss = beta_s * (q_s - v_s).detach()   # [B, 1]
        # https://github.com/lweitkamp/option-critic-pytorch/blob/0c57da7686f8903ed2d8dded3fae832ee9defd1a/option_critic.py#L238
        if self.terminal_mask:
            beta_loss *= (1 - BATCH.done)
        beta_loss = beta_loss.mean()  # [B, 1] => 1

        self.intra_option_oplr.step(pi_loss)
        self.termination_oplr.step(beta_loss)

        self.global_step.add_(1)
        return td_error, dict([
            ['LOSS/q_loss', q_loss.mean()],
            ['LOSS/pi_loss', pi_loss.mean()],
            ['LOSS/beta_loss', beta_loss.mean()],
            ['Statistics/q_option_max', q_s.max()],
            ['Statistics/q_option_min', q_s.min()],
            ['Statistics/q_option_mean', q_s.mean()]
        ])

    def store_data(self, exps: BatchExperiences):
        # self._running_average()
        self.data.add(OC_BatchExperiences(*exps.astuple(), self.last_options, self.options))

    def prefill_store(self, exps: BatchExperiences):
        pass
