#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from typing import (Union,
                    List,
                    NoReturn)

from rls.algorithms.base.off_policy import Off_Policy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import (sync_params_pairs,
                                   q_target_func)
from rls.nn.models import CriticQvalueAll
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


class DQN(Off_Policy):
    '''
    Deep Q-learning Network, DQN, [2013](https://arxiv.org/pdf/1312.5602.pdf), [2015](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    DQN + LSTM, https://arxiv.org/abs/1507.06527
    '''

    def __init__(self,
                 envspec,

                 lr: float = 5.0e-4,
                 eps_init: float = 1,
                 eps_mid: float = 0.2,
                 eps_final: float = 0.01,
                 init2mid_annealing_step: int = 1000,
                 assign_interval: int = 1000,
                 network_settings: List[int] = [32, 32],
                 **kwargs):
        assert not envspec.is_continuous, 'dqn only support discrete action space'
        super().__init__(envspec=envspec, **kwargs)
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval

        self.q_net = CriticQvalueAll(self.rep_net.h_dim, output_shape=self.a_dim, network_settings=network_settings).to(self.device)
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

    def __call__(self, obs, evaluation: bool = False) -> np.ndarray:
        if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
            actions = np.random.randint(0, self.a_dim, self.n_copys)
        else:
            actions, self.cell_state = self.call(obs, cell_state=self.cell_state)
        return actions

    @iTensor_oNumpy
    def call(self, obs, cell_state):
        feat, cell_state = self.rep_net(obs, cell_state=cell_state)
        q_values = self.q_net(feat)
        a = q_values.argmax(1)
        return a, cell_state

    def _target_params_update(self):
        if self.global_step % self.assign_interval == 0:
            sync_params_pairs(self._pairs)

    def learn(self, **kwargs) -> NoReturn:
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([['LEARNING_RATE/lr', self.oplr.lr]])
            })

    @iTensor_oNumpy
    def _train(self, BATCH, isw, cell_states):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
        feat_, _ = self._target_rep_net(BATCH.obs_, cell_state=cell_states['obs_'])
        q = self.q_net(feat)
        q_next = self.q_target_net(feat_)
        q_eval = (q * BATCH.action).sum(1, keepdim=True)
        q_target = q_target_func(BATCH.reward,
                                 self.gamma,
                                 BATCH.done,
                                 q_next.max(1, keepdim=True)[0])
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

    # @iTensor_oNumpy
    # def _cal_td(self, BATCH, cell_states):
        # feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
        # feat_, _ = self._target_rep_net(BATCH.obs_, cell_state=cell_states['obs_'])
        # q = self.q_net(feat)
        # q_next = self.q_target_net(feat_)
        # q_eval = (q * BATCH.action).sum(1, keepdim=True)
        # q_target = q_target_func(BATCH.reward,
        #  self.gamma,
        #  BATCH.done,
        #  q_next.max(1, keepdim=True)[0])
        # td_error = q_target - q_eval
    #     return td_error

    # def apex_learn(self, train_step, data, priorities):
    #     self.train_step = train_step
    #     return self._apex_learn(function_dict={
    #         'summary_dict': dict([['LEARNING_RATE/lr', self.oplr.lr]])
    #     }, data=data, priorities=priorities)

    # def apex_cal_td(self, data):
    #     return self._apex_cal_td(data=data)
