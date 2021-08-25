#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from torch import distributions as td

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import q_target_func
from rls.nn.models import CriticQvalueBootstrap
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy
from rls.nn.modules.wrappers import TargetTwin
from rls.common.specs import Data


class BootstrappedDQN(SarlOffPolicy):
    '''
    Deep Exploration via Bootstrapped DQN, http://arxiv.org/abs/1602.04621
    '''
    policy_mode = 'off-policy'

    def __init__(self,
                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=1000,
                 head_num=4,
                 network_settings=[32, 32],
                 **kwargs):
        super().__init__(**kwargs)
        assert not self.is_continuous, 'Bootstrapped DQN only support discrete action space'
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval
        self.head_num = head_num
        self._probs = t.FloatTensor([1. / head_num for _ in range(head_num)])
        self.now_head = 0

        self.q_net = TargetTwin(CriticQvalueBootstrap(self.obs_spec,
                                                      rep_net_params=self.rep_net_params,
                                                      output_shape=self.a_dim,
                                                      head_num=self.head_num,
                                                      network_settings=network_settings)).to(self.device)

        self.oplr = OPLR(self.q_net, lr)
        self._trainer_modules.update(model=self.q_net,
                                     oplr=self.oplr)

    def episode_reset(self):
        super().episode_reset()
        self.now_head = np.random.randint(self.head_num)

    @iTensor_oNumpy
    def __call__(self, obs):
        if self._is_train_mode and self.expl_expt_mng.is_random(self.cur_train_step):
            actions = np.random.randint(0, self.a_dim, self.n_copys)
        else:
            q_values = self.q_net(obs, cell_state=self.cell_state)  # [H, B, A]
            self.next_cell_state = self.q_net.get_cell_state()
            actions = q_values[self.now_head].argmax(-1)  # [H, B, A] => [B, A] => [B, ]
        return Data(action=actions)

    @iTensor_oNumpy
    def _train(self, BATCH):
        q = self.q_net(BATCH.obs).mean(0)   # [H, T, B, A] => [T, B, A]
        q_next = self.q_net.t(BATCH.obs_).mean(0)    # [H, T, B, A] => [T, B, A]
        q_eval = (q * BATCH.action).sum(-1, keepdim=True)    # [T, B, A] * [T, B, A] => [T, B, 1]
        q_target = q_target_func(BATCH.reward,
                                 self.gamma,
                                 BATCH.done,
                                 q_next.max(-1, keepdim=True)[0],    # [T, B, A] => [T, B, 1]
                                 BATCH.begin_mask,
                                 use_rnn=self.use_rnn)  # [T, B, 1]
        td_error = q_target - q_eval    # [T, B, 1]
        q_loss = (td_error.square()*BATCH.get('isw', 1.0)).mean()   # 1

        # mask_dist = td.Bernoulli(probs=self._probs)  # TODO:
        # mask = mask_dist.sample([batch_size]).T   # [H, B]
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
            self.q_net.sync()
