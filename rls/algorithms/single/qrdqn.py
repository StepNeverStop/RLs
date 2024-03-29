#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as th
import torch.nn.functional as F

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.common.data import Data
from rls.common.decorator import iton
from rls.nn.models import QrdqnDistributional
from rls.nn.modules.wrappers import TargetTwin
from rls.nn.utils import OPLR
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import n_step_return


class QRDQN(SarlOffPolicy):
    """
    Quantile Regression DQN
    Distributional Reinforcement Learning with Quantile Regression, https://arxiv.org/abs/1710.10044
    No double, no dueling, no noisy net.
    """
    policy_mode = 'off-policy'

    def __init__(self,
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
        assert nums > 0, 'assert nums > 0'
        super().__init__(**kwargs)
        assert not self.is_continuous, 'qrdqn only support discrete action space'
        self.nums = nums
        self.huber_delta = huber_delta
        self.quantiles = th.tensor((2 * np.arange(self.nums) + 1) / (2.0 * self.nums)).float().to(self.device)  # [N,]
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self._max_train_step)
        self.assign_interval = assign_interval
        self.q_net = TargetTwin(QrdqnDistributional(self.obs_spec,
                                                    rep_net_params=self._rep_net_params,
                                                    action_dim=self.a_dim,
                                                    nums=self.nums,
                                                    network_settings=network_settings)).to(self.device)
        self.oplr = OPLR(self.q_net, lr, **self._oplr_params)
        self._trainer_modules.update(model=self.q_net,
                                     oplr=self.oplr)

    @iton
    def select_action(self, obs):
        q_values = self.q_net(obs, rnncs=self.rnncs)  # [B, A, N]
        self.rnncs_ = self.q_net.get_rnncs()

        if self._is_train_mode and self.expl_expt_mng.is_random(self._cur_train_step):
            actions = np.random.randint(0, self.a_dim, self.n_copies)
        else:
            q = q_values.mean(-1)  # [B, A, N] => [B, A]
            actions = q.argmax(-1)  # [B,]
        return actions, Data(action=actions)

    @iton
    def _train(self, BATCH):
        q_dist = self.q_net(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A, N]
        q_dist = (q_dist * BATCH.action.unsqueeze(-1)).sum(-2)  # [T, B, A, N] => [T, B, N]

        target_q_dist = self.q_net.t(BATCH.obs_, begin_mask=BATCH.begin_mask)  # [T, B, A, N]
        target_q = target_q_dist.mean(-1)  # [T, B, A, N] => [T, B, A]
        _a = target_q.argmax(-1)  # [T, B]
        next_max_action = F.one_hot(_a, self.a_dim).float().unsqueeze(-1)  # [T, B, A, 1]
        # [T, B, A, N] => [T, B, N]
        target_q_dist = (target_q_dist * next_max_action).sum(-2)

        target = n_step_return(BATCH.reward.repeat(1, 1, self.nums),
                               self.gamma,
                               BATCH.done.repeat(1, 1, self.nums),
                               target_q_dist,
                               BATCH.begin_mask.repeat(1, 1, self.nums)).detach()  # [T, B, N]

        q_eval = q_dist.mean(-1, keepdim=True)  # [T, B, 1]
        q_target = target.mean(-1, keepdim=True)  # [T, B, 1]
        td_error = q_target - q_eval  # [T, B, 1], used for PER

        target = target.unsqueeze(-2)  # [T, B, 1, N]
        q_dist = q_dist.unsqueeze(-1)  # [T, B, N, 1]

        # [T, B, 1, N] - [T, B, N, 1] => [T, B, N, N]
        quantile_error = target - q_dist
        huber = F.huber_loss(target, q_dist, reduction="none", delta=self.huber_delta)  # [T, B, N, N]
        # [N,] - [T, B, N, N] => [T, B, N, N]
        huber_abs = (self.quantiles - quantile_error.detach().le(0.).float()).abs()
        loss = (huber_abs * huber).mean(-1)  # [T, B, N, N] => [T, B, N]
        loss = loss.sum(-1, keepdim=True)  # [T, B, N] => [T, B, 1]
        loss = (loss * BATCH.get('isw', 1.0)).mean()  # 1

        self.oplr.optimize(loss)
        return td_error, {
            'LEARNING_RATE/lr': self.oplr.lr,
            'LOSS/loss': loss,
            'Statistics/q_max': q_eval.max(),
            'Statistics/q_min': q_eval.min(),
            'Statistics/q_mean': q_eval.mean()
        }

    def _after_train(self):
        super()._after_train()
        if self._cur_train_step % self.assign_interval == 0:
            self.q_net.sync()
