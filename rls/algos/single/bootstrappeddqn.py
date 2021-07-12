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
from rls.nn.models import CriticQvalueBootstrap
from rls.nn.utils import OPLR
from rls.utils.converter import to_numpy
from rls.common.decorator import iTensor_oNumpy


class BootstrappedDQN(Off_Policy):
    '''
    Deep Exploration via Bootstrapped DQN, http://arxiv.org/abs/1602.04621
    '''

    def __init__(self,
                 envspec,

                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=1000,
                 head_num=4,
                 network_settings=[32, 32],
                 **kwargs):
        assert not envspec.is_continuous, 'Bootstrapped DQN only support discrete action space'
        super().__init__(envspec=envspec, **kwargs)
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval
        self.head_num = head_num
        self._probs = [1. / head_num for _ in range(head_num)]
        self.now_head = 0

        self.q_net = CriticQvalueBootstrap(self.rep_net.h_dim,
                                           output_shape=self.a_dim,
                                           head_num=self.head_num,
                                           network_settings=network_settings).to(self.device)
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

    def reset(self):
        super().reset()
        self.now_head = np.random.randint(self.head_num)

    @iTensor_oNumpy
    def __call__(self, obs, evaluation=False):
        if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
            a = np.random.randint(0, self.a_dim, self.n_copys)
        else:
            feat, self.cell_state = self.rep_net(obs, cell_state=self.cell_state)
            q_values = self.q_net(feat)  # [H, B, A]
            q_values = to_numpy(q_values)
            a = np.argmax(q_values[self.now_head], axis=1)  # [H, B, A] => [B, A] => [B, ]
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
        batch_size = BATCH.action.shape[0]
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
        q = self.q_net(feat)   # [H, B, A]
        feat_, _ = self._target_rep_net(BATCH.obs_, cell_state=cell_states['obs_'])
        q_next = self.q_target_net(feat_)   # [H, B, A]
        q_eval = (q * BATCH.action).sum(-1, keepdim=True)    # [H, B, A] * [B, A] => [H, B, 1]
        q_target = q_target_func(BATCH.reward,
                                 self.gamma,
                                 BATCH.done,
                                 q_next.max(-1, keepdim=True)[0])
        td_error = q_target - q_eval    # [H, B, 1]
        td_error = td_error.sum(-1)  # [H, B]

        mask_dist = td.bernoulli.Bernoulli(probs=self._probs)
        mask = mask_dist.sample([batch_size]).T   # [H, B]
        q_loss = (td_error.square() * isw).mean()

        self.oplr.step(q_loss)
        self.global_step.add_(1)
        return td_error.mean(0), dict([  # [H, B] =>
            ['LOSS/loss', q_loss],
            ['Statistics/q_max', q_eval.max()],
            ['Statistics/q_min', q_eval.min()],
            ['Statistics/q_mean', q_eval.mean()]
        ])
