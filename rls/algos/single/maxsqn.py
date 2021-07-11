#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from torch import distributions as td

from rls.algos.base.off_policy import Off_Policy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import (sync_params_pairs,
                                   q_target_func)
from rls.nn.models import CriticQvalueAll
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


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
        self.log_alpha = alpha if not auto_adaption else t.tensor(0., requires_grad=True)
        self.auto_adaption = auto_adaption
        self.target_entropy = beta * np.log(self.a_dim)

        self.critic = CriticQvalueAll(self.rep_net.h_dim,
                                      output_shape=self.a_dim,
                                      network_settings=network_settings)
        self.critic2 = CriticQvalueAll(self.rep_net.h_dim,
                                       output_shape=self.a_dim,
                                       network_settings=network_settings)

        self._target_rep_net = deepcopy(self.rep_net)
        self._target_rep_net.eval()
        self.critic_target = deepcopy(self.critic)
        self.critic_target.eval()
        self.critic2_target = deepcopy(self.critic2)
        self.critic2_target.eval()

        self._pairs = [(self.critic_target, self.critic)
                       (self.critic2_target, self.critic2),
                       (self._target_rep_net, self.rep_net)]
        sync_params_pairs(self._pairs)

        self.critic_oplr = OPLR([self.critic, self.critic2, self.rep_net], q_lr)
        self.alpha_oplr = OPLR(self.log_alpha, alpha_lr)

        self._worker_modules.update(rep_net=self.rep_net,
                                    critic=self.critic)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(critic2=self.critic2,
                                     critic_oplr=self.critic_oplr,
                                     alpha_oplr=self.alpha_oplr)
        self.initialize_data_buffer()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def __call__(self, obs, evaluation=False):
        if self.use_epsilon and np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
            a = np.random.randint(0, self.a_dim, self.n_copys)
        else:
            mu, pi = self._get_action(obs)
            a = pi
        return a

    @iTensor_oNumpy
    def _get_action(self, obs):
        feat, self.cell_state = self.rep_net(obs, cell_state=self.cell_state)
        q = self.critic(feat)
        cate_dist = td.categorical.Categorical(logits=(q / self.alpha))
        pi = cate_dist.sample()
        return q.argmax(1), pi

    def _target_params_update(self):
        sync_params_pairs(self._pairs, self.ployak)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([
                    ['LEARNING_RATE/critic_lr', self.critic_oplr.lr],
                    ['LEARNING_RATE/alpha_lr', self.alpha_oplr.lr]
                ])
            })

    @iTensor_oNumpy
    def _train(self, BATCH, isw, cell_states):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
        feat_, _ = self._target_rep_net(BATCH.obs_, cell_state=cell_states['obs_'])
        q1 = self.critic(feat)
        q2 = self.critic2(feat)
        q1_eval = (q1 * BATCH.action).sum(1, keepdim=True)
        q2_eval = (q2 * BATCH.action).sum(1, keepdim=True)

        ret = self.critic_target(BATCH.obs_, cell_state=cell_states['obs_'])
        q1_target = self.critic_target(feat_)
        q1_target = self.critic2_target(feat_)
        q1_target_max = q1_target.max(1, keepdim=True)[0]
        q1_target_log_probs = (q1_target / (self.alpha + 1e-8)).log_softmax(-1)
        q1_target_entropy = -(q1_target_log_probs.exp() * q1_target_log_probs).sum(1, keepdim=True).mean()

        q2_target_max = q2_target.max(1, keepdim=True)[0]
        # q2_target_log_probs = q2_target.log_softmax(-1)
        # q2_target_log_max = q2_target_log_probs.max(1, keepdim=True)[0]

        q_target = t.minimum(q1_target_max, q2_target_max) + self.alpha * q1_target_entropy
        dc_r = q_target_func(BATCH.reward,
                             self.gamma,
                             BATCH.done,
                             q_target)
        td_error1 = q1_eval - dc_r
        td_error2 = q2_eval - dc_r
        q1_loss = (td_error1.square() * isw).mean()
        q2_loss = (td_error2.square() * isw).mean()
        loss = 0.5 * (q1_loss + q2_loss)
        self.critic_oplr.step(loss)
        summaries = dict([
            ['LOSS/loss', loss],
            ['Statistics/log_alpha', self.log_alpha],
            ['Statistics/alpha', self.alpha],
            ['Statistics/q1_entropy', q1_entropy],
            ['Statistics/q_min', t.minimum(q1, q2).mean()],
            ['Statistics/q_mean', q1.mean()],
            ['Statistics/q_max', t.maximum(q1, q2).mean()]
        ])
        if self.auto_adaption:
            q1_log_probs = (q1 / (self.alpha + 1e-8)).log_softmax(-1)
            q1_entropy = -(q1_log_probs.exp() * q1_log_probs).sum(1, keepdim=True).mean()
            alpha_loss = -(self.alpha * (self.target_entropy - q1_entropy).detach()).mean()
            self.alpha_oplr.step(alpha_loss)

            summaries.update({
                'LOSS/alpha_loss': alpha_loss
            })
        self.global_step.add_(1)
        return (td_error1 + td_error2) / 2, summaries
