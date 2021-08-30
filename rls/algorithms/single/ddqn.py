#!/usr/bin/env python3
# encoding: utf-8

from typing import List, NoReturn, Union

import numpy as np
import torch as t

from rls.algorithms.single.dqn import DQN
from rls.common.decorator import iTensor_oNumpy
from rls.utils.torch_utils import n_step_return


class DDQN(DQN):
    '''
    Double DQN, https://arxiv.org/abs/1509.06461
    Double DQN + LSTM, https://arxiv.org/abs/1908.06040
    '''
    policy_mode = 'off-policy'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @iTensor_oNumpy
    def _train(self, BATCH):
        q = self.q_net(BATCH.obs, begin_mask=BATCH.begin_mask)   # [T, B, A]
        q_next = self.q_net(
            BATCH.obs_, begin_mask=BATCH.begin_mask)  # [T, B, A]
        q_target_next = self.q_net.t(
            BATCH.obs_, begin_mask=BATCH.begin_mask)    # [T, B, A]
        next_max_action = q_next.argmax(-1)  # [T, B]
        next_max_action_one_hot = t.nn.functional.one_hot(
            next_max_action.squeeze(), self.a_dim).float()  # [T, B, A]
        q_eval = (q * BATCH.action).sum(-1, keepdim=True)    # [T, B, 1]
        q_target_next_max = (
            q_target_next * next_max_action_one_hot).sum(-1, keepdim=True)  # [T, B, 1]
        q_target = n_step_return(BATCH.reward,
                                 self.gamma,
                                 BATCH.done,
                                 q_target_next_max,
                                 BATCH.begin_mask).detach()  # [T, B, 1]
        td_error = q_target - q_eval    # [T, B, 1]
        q_loss = (td_error.square()*BATCH.get('isw', 1.0)).mean()   # 1
        self.oplr.step(q_loss)
        return td_error, dict([
            ['LEARNING_RATE/lr', self.oplr.lr],
            ['LOSS/loss', q_loss],
            ['Statistics/q_max', q_eval.max()],
            ['Statistics/q_min', q_eval.min()],
            ['Statistics/q_mean', q_eval.mean()]
        ])
