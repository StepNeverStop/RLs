#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from typing import (Union,
                    List,
                    NoReturn)

from rls.algorithms.single.dqn import DQN
from rls.utils.torch_utils import q_target_func
from rls.common.decorator import iTensor_oNumpy


class DDQN(DQN):
    '''
    Double DQN, https://arxiv.org/abs/1509.06461
    Double DQN + LSTM, https://arxiv.org/abs/1908.06040
    '''

    def __init__(self,
                 envspec,

                 lr: float = 5.0e-4,
                 eps_init: float = 1,
                 eps_mid: float = 0.2,
                 eps_final: float = 0.01,
                 init2mid_annealing_step: int = 1000,
                 assign_interval: int = 2,
                 network_settings: List = [32, 32],
                 **kwargs):
        assert not envspec.is_continuous, 'double dqn only support discrete action space'
        super().__init__(
            envspec=envspec,
            lr=lr,
            eps_init=eps_init,
            eps_mid=eps_mid,
            eps_final=eps_final,
            init2mid_annealing_step=init2mid_annealing_step,
            assign_interval=assign_interval,
            network_settings=network_settings,
            **kwargs)

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
        feat__, _ = self.rep_net(BATCH.obs_, cell_state=cell_states['obs_'])
        q = self.q_net(feat)
        q_next = self.q_net(feat__)
        q_target_next = self.q_target_net(feat_)
        next_max_action = q_next.argmax(1)
        next_max_action_one_hot = t.nn.functional.one_hot(next_max_action.squeeze(), self.a_dim).float()
        q_eval = (q * BATCH.action).sum(1, keepdim=True)
        q_target_next_max = (q_target_next * next_max_action_one_hot).sum(1, keepdim=True)
        q_target = q_target_func(BATCH.reward,
                                 self.gamma,
                                 BATCH.done,
                                 q_target_next_max)
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
