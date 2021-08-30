#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t
from torch import distributions as td

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.common.decorator import iTensor_oNumpy
from rls.common.specs import Data
from rls.nn.models import CriticQvalueAll
from rls.nn.modules.wrappers import TargetTwin
from rls.nn.utils import OPLR
from rls.utils.torch_utils import n_step_return


class SQL(SarlOffPolicy):
    '''
        Soft Q-Learning. ref: https://github.com/Bigpig4396/PyTorch-Soft-Q-Learning/blob/master/SoftQ.py
        NOTE: not the original of the paper, NO SVGD.
        Reinforcement Learning with Deep Energy-Based Policies: https://arxiv.org/abs/1702.08165
    '''
    policy_mode = 'off-policy'

    def __init__(self,
                 lr=5.0e-4,
                 alpha=2,
                 ployak=0.995,
                 network_settings=[32, 32],
                 **kwargs):
        super().__init__(**kwargs)
        assert not self.is_continuous, 'sql only support discrete action space'
        self.alpha = alpha
        self.ployak = ployak

        self.q_net = TargetTwin(CriticQvalueAll(self.obs_spec,
                                                rep_net_params=self._rep_net_params,
                                                output_shape=self.a_dim,
                                                network_settings=network_settings),
                                self.ployak).to(self.device)

        self.oplr = OPLR(self.q_net, lr)
        self._trainer_modules.update(model=self.q_net,
                                     oplr=self.oplr)

    @iTensor_oNumpy
    def select_action(self, obs):
        q_values = self.q_net(obs, cell_state=self.cell_state)  # [B, A]
        self.next_cell_state = self.q_net.get_cell_state()
        logits = ((q_values - self._get_v(q_values)) /
                  self.alpha).exp()    # > 0   # [B, A]
        logits /= logits.sum(-1, keepdim=True)  # [B, A]
        cate_dist = td.Categorical(logits=logits)
        actions = pi = cate_dist.sample()    # [B,]
        return actions, Data(action=actions)

    def _get_v(self, q):
        v = self.alpha * (q / self.alpha).exp().mean(-1,
                                                     keepdim=True).log()    # [B, 1] or [T, B, 1]
        return v

    @iTensor_oNumpy
    def _train(self, BATCH):
        q = self.q_net(BATCH.obs, begin_mask=BATCH.begin_mask)   # [T, B, A]
        q_next = self.q_net.t(
            BATCH.obs_, begin_mask=BATCH.begin_mask)    # [T, B, A]
        v_next = self._get_v(q_next)     # [T, B, 1]
        q_eval = (q * BATCH.action).sum(-1, keepdim=True)    # [T, B, 1]
        q_target = n_step_return(BATCH.reward,
                                 self.gamma,
                                 BATCH.done,
                                 v_next,
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

    def _after_train(self):
        super()._after_train()
        self.q_net.sync()
