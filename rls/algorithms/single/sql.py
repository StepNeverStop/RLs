#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from torch import distributions as td

from rls.algorithms.base.off_policy import Off_Policy
from rls.utils.torch_utils import (sync_params_list,
                                   q_target_func)
from rls.nn.models import CriticQvalueAll
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


class SQL(Off_Policy):
    '''
        Soft Q-Learning. ref: https://github.com/Bigpig4396/PyTorch-Soft-Q-Learning/blob/master/SoftQ.py
        NOTE: not the original of the paper, NO SVGD.
        Reinforcement Learning with Deep Energy-Based Policies: https://arxiv.org/abs/1702.08165
    '''

    def __init__(self,
                 envspec,

                 lr=5.0e-4,
                 alpha=2,
                 ployak=0.995,
                 network_settings=[32, 32],
                 **kwargs):
        assert not envspec.is_continuous, 'sql only support discrete action space'
        super().__init__(envspec=envspec, **kwargs)
        self.alpha = alpha
        self.ployak = ployak

        self.q_net = CriticQvalueAll(self.rep_net.h_dim,
                                     output_shape=self.a_dim,
                                     network_settings=network_settings).to(self.device)
        self.q_target_net = deepcopy(self.q_net)
        self.q_target_net.eval()

        self._target_rep_net = deepcopy(self.rep_net)
        self._target_rep_net.eval()

        self._pairs = [(self.q_target_net, self._target_rep_net),
                       (self.q_net, self.rep_net)]
        sync_params_list(self._pairs)

        self.oplr = OPLR([self.q_net, self.rep_net], lr)

        self._worker_modules.update(rep_net=self.rep_net,
                                    model=self.q_net)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(oplr=self.oplr)
        self.initialize_data_buffer()

    def __call__(self, obs, evaluation=False):
        actions, self.cell_state = self.call(obs, cell_state=self.cell_state)
        return actions

    @iTensor_oNumpy
    def call(self, obs, cell_state):
        feat, cell_state = self.rep_net(obs, cell_state=cell_state)
        q_values = self.q_net(feat)
        logits = ((q_values - self.get_v(q_values)) / self.alpha).exp()    # > 0
        logits /= logits.sum()
        cate_dist = td.Categorical(logits=logits)
        pi = cate_dist.sample()
        return pi, cell_state

    def get_v(self, q):
        v = self.alpha * (q / self.alpha).exp().mean(1, keepdim=True).log()
        return v

    def _target_params_update(self):
        sync_params_list(self._pairs, self.ployak)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([['LEARNING_RATE/lr', self.oplr.lr]])
            })

    @iTensor_oNumpy
    def _train(self, BATCH, isw, cell_states):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
        q = self.q_net(feat)
        feat_, _ = self._target_rep_net(BATCH.obs_, cell_state=cell_states['obs_'])
        q_next = self.q_target_net(feat_)
        v_next = self.get_v(q_next)
        q_eval = (q * BATCH.action).sum(1, keepdim=True)
        q_target = q_target_func(BATCH.reward,
                                 self.gamma,
                                 BATCH.done,
                                 v_next)
        td_error = q_target - q_eval
        q_loss = (td_error.square() * isw).mean()
        self.oplr.step(q_loss)
        self.global_step.add_(1)
        return td_error, dict([
            ['LOSS/loss', q_loss],
            ['Statistics/q_max', q_eval.max()],
            ['Statistics/q_min', q_eval.min()],
            ['Statistics/q_mean', q_eval.mean()]
        ])
