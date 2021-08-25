#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from typing import (Union,
                    List,
                    NoReturn)

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import (sync_params,
                                   q_target_func)
from rls.nn.models import CriticQvalueAll
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy
from rls.common.specs import Data


class AveragedDQN(SarlOffPolicy):
    '''
    Averaged-DQN, http://arxiv.org/abs/1611.01929
    '''
    policy_mode = 'off-policy'

    def __init__(self,
                 target_k: int = 4,
                 lr: float = 5.0e-4,
                 eps_init: float = 1,
                 eps_mid: float = 0.2,
                 eps_final: float = 0.01,
                 init2mid_annealing_step: int = 1000,
                 assign_interval: int = 1000,
                 network_settings: List[int] = [32, 32],
                 **kwargs):
        super().__init__(**kwargs)
        assert not self.is_continuous, 'dqn only support discrete action space'
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval
        self.target_k = target_k
        assert self.target_k > 0, "assert self.target_k > 0"
        self.current_target_idx = 0

        self.q_net = CriticQvalueAll(self.obs_spec,
                                     rep_net_params=self.rep_net_params,
                                     output_shape=self.a_dim,
                                     network_settings=network_settings).to(self.device)
        self.target_nets = []
        for i in range(self.target_k):
            target_q_net = deepcopy(self.q_net)
            target_q_net.eval()
            sync_params(target_q_net, self.q_net)
            self.target_nets.append(target_q_net)

        self.oplr = OPLR(self.q_net, lr)
        self._trainer_modules.update(model=self.q_net,
                                     oplr=self.oplr)

    @iTensor_oNumpy
    def __call__(self, obs):
        if self._is_train_mode and self.expl_expt_mng.is_random(self.cur_train_step):
            actions = np.random.randint(0, self.a_dim, self.n_copys)
        else:
            q_values = self.q_net(obs, cell_state=self.cell_state)  # [B, *]
            self.next_cell_state = self.q_net.get_cell_state()
            for i in range(self.target_k):
                target_q_values = self.target_nets[i](obs, cell_state=self.cell_state)
                q_values += target_q_values
            actions = q_values.argmax(-1)  # 不取平均也可以 [B, ]
        return Data(action=actions)

    @iTensor_oNumpy
    def _train(self, BATCH):
        q = self.q_net(BATCH.obs)   # [T, B, *]
        q_next = 0
        for i in range(self.target_k):
            q_next += self.target_nets[i](BATCH.obs_)
        q_next /= self.target_k  # [T, B, *]
        q_eval = (q * BATCH.action).sum(-1, keepdim=True)     # [T, B, 1]
        q_target = q_target_func(BATCH.reward,
                                 self.gamma,
                                 BATCH.done,
                                 q_next.max(-1, keepdim=True)[0],
                                 BATCH.begin_mask,
                                 use_rnn=self.use_rnn)    # [T, B, 1]
        td_error = q_target - q_eval      # [T, B, 1]
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
        if self.cur_train_step % self.assign_interval == 0:
            sync_params(self.target_nets[self.current_target_idx], self.q_net)
            self.current_target_idx = (self.current_target_idx + 1) % self.target_k
