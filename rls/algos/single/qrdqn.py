#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy

from rls.algos.base.off_policy import Off_Policy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import (huber_loss,
                                   sync_params_pairs)
from rls.nn.models import QrdqnDistributional
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


class QRDQN(Off_Policy):
    '''
    Quantile Regression DQN
    Distributional Reinforcement Learning with Quantile Regression, https://arxiv.org/abs/1710.10044
    No double, no dueling, no noisy net.
    '''

    def __init__(self,
                 envspec,

                 nums=20,
                 huber_delta=1.,
                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=1000,
                 network_settings=[128, 128],
                 **kwargs):
        assert not envspec.is_continuous, 'qrdqn only support discrete action space'
        assert nums > 0
        super().__init__(envspec=envspec, **kwargs)
        self.nums = nums
        self.huber_delta = huber_delta
        self.quantiles = t.tensor((2 * np.arange(self.nums) + 1) / (2.0 * self.nums)).float().view(-1, self.nums)  # [1, N]
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval

        self.q_net = QrdqnDistributional(self.rep_net.h_dim,
                                         action_dim=self.a_dim,
                                         nums=self.nums,
                                         network_settings=network_settings)
        self.q_target_net = deepcopy(self.q_net)
        self.q_target_net.eval()

        self._target_rep_net = deepcopy(self.rep_net)
        self._target_rep_net.eval()

        self._pairs = [(self.q_target_net, self.q_net),
                       (self._target_rep_net, self.rep_net)]
        sync_params_pairs(self._pairs)

        self.oplr = OPLR([self.q_net, self.rep_net], lr)

        self._worker_modules.update(rep_net=self.rep_net,
                                    model=self.q_net)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(oplr=self.oplr)
        self.initialize_data_buffer()

    @iTensor_oNumpy
    def __call__(self, obs, evaluation=False):
        if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
            a = np.random.randint(0, self.a_dim, self.n_copys)
        else:
            feat, self.cell_state = self.rep_net(obs, cell_state=self.cell_state)
            q_values = self.q_net(feat)
            q = q_values.mean(-1)  # [B, A, N] => [B, A]
            a = q.argmax(-1)  # [B, 1]
        return a

    def _target_params_update(self):
        if self.global_step % self.assign_interval == 0:
            sync_params_pairs(self._pairs)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([['LEARNING_RATE/lr', self.oplr.lr]])
            })

    @iTensor_oNumpy
    def _train(self, BATCH, isw, cell_states):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
        feat_, _ = self._target_rep_net(BATCH.obs_, cell_state=cell_states['obs_'])
        batch_size = BATCH.action.shape[0]
        indexes = t.arange(batch_size).view(-1, 1)  # [B, 1]
        q_dist = self.q_net(feat)  # [B, A, N]
        q_dist = (q_dist.permute(2, 0, 1) * BATCH.action).sum(-1).T  # [B, N]

        target_q_dist = self.q_target_net(feat_)  # [B, A, N]
        target_q = target_q_dist.mean(-1)  # [B, A, N] => [B, A]
        a_ = target_q.argmax(-1).view(-1, 1)  # [B, 1]
        target_q_dist = target_q_dist[list(t.cat([indexes, a_], -1).long().T)]   # [B, N]
        target = BATCH.reward.repeat(1, self.nums) \
            + self.gamma * target_q_dist * (1.0 - BATCH.done.repeat(1, self.nums))  # [B, N], [B, N]* [B, N] = [B, N]

        q_eval = q_dist.mean(-1)    # [B, 1]
        q_target = target.mean(-1)  # [B, 1]
        td_error = q_target - q_eval     # [B, 1], used for PER

        quantile_error = target.unsqueeze(1) - q_dist.unsqueeze(-1)   # [B, 1, N] - [B, N, 1] => [B, N, N]
        huber = huber_loss(quantile_error, delta=self.huber_delta)  # [B, N, N]
        huber_abs = (self.quantiles - t.where(quantile_error < 0, 1., 0.)).abs()   # [1, N] - [B, N, N] => [B, N, N]
        loss = (huber_abs * huber).mean(-1)  # [B, N, N] => [B, N]
        loss = loss.sum(-1)  # [B, N] => [B, ]
        loss = (loss * isw).mean()  # [B, ] => 1

        self.oplr.step(loss)
        self.global_step.add_(1)
        return td_error, dict([
            ['LOSS/loss', loss],
            ['Statistics/q_max', q_eval.max()],
            ['Statistics/q_min', q_eval.min()],
            ['Statistics/q_mean', q_eval.mean()]
        ])
