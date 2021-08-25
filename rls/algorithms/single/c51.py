#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.nn.models import C51Distributional
from rls.utils.torch_utils import q_target_func
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy
from rls.nn.modules.wrappers import TargetTwin
from rls.common.specs import Data


class C51(SarlOffPolicy):
    '''
    Category 51, https://arxiv.org/abs/1707.06887
    No double, no dueling, no noisy net.
    '''
    policy_mode = 'off-policy'

    def __init__(self,
                 v_min=-10,
                 v_max=10,
                 atoms=51,
                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=1000,
                 network_settings=[128, 128],
                 **kwargs):
        super().__init__(**kwargs)
        assert not self.is_continuous, 'c51 only support discrete action space'
        self.v_min = v_min
        self.v_max = v_max
        self.atoms = atoms
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self.z = t.tensor([self.v_min + i * self.delta_z for i in range(self.atoms)]).float().to(self.device)  # [N,]
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval
        self.q_net = TargetTwin(C51Distributional(self.obs_spec,
                                                  rep_net_params=self.rep_net_params,
                                                  action_dim=self.a_dim,
                                                  atoms=self.atoms,
                                                  network_settings=network_settings)).to(self.device)
        self.oplr = OPLR(self.q_net, lr)
        self._trainer_modules.update(model=self.q_net,
                                     oplr=self.oplr)

    @iTensor_oNumpy
    def __call__(self, obs):
        if self._is_train_mode and self.expl_expt_mng.is_random(self.cur_train_step):
            actions = np.random.randint(0, self.a_dim, self.n_copys)
        else:
            feat = self.q_net(obs, cell_state=self.cell_state)  # [B, N, A]
            self.next_cell_state = self.q_net.get_cell_state()
            feat = feat.swapaxes(-1, -2)  # [B, N, A] => [B, A, N]
            q = (self.z * feat).sum(-1)  # [B, A, N] * [N,] => [B, A]
            actions = q.argmax(-1)  # [B,]
        return Data(action=actions)

    @iTensor_oNumpy
    def _train(self, BATCH):
        time_step = BATCH.reward.shape[0]
        batch_size = BATCH.reward.shape[1]
        indexes = t.arange(time_step*batch_size).view(-1, 1)  # [T*B, 1]

        q_dist = self.q_net(BATCH.obs)  # [T, B, N, A]
        q_dist = q_dist.permute(2, 0, 1, 3)  # [T, B, N, A] => [N, T, B, A]
        q_dist = (q_dist * BATCH.action).sum(-1)  # [N, T, B, A] => [N, T, B]
        q_dist = q_dist.permute(1, 2, 0)  # [N, T, B] => [T, B, N]
        q_eval = (q_dist * self.z).sum(-1)  # [T, B, N] * [N,] => [T, B]
        q_dist = q_dist.view(-1, self.atoms)  # [T, B, N] => [T*B, N]

        target_q_dist = self.q_net.t(BATCH.obs)  # [T, B, N, A]
        target_q = (target_q_dist.swapaxes(-1, -2) * self.z).sum(-1)  # [T, B, N, A] => [T, B, A, N] * [1, N] => [T, B, A]
        a_ = target_q.argmax(-1)  # [T, B]
        a_onehot = t.nn.functional.one_hot(a_, self.a_dim).float()  # [T, B, A]
        target_q_dist = target_q_dist.permute(2, 0, 1, 3)  # [T, B, N, A] => [N, T, B, A]
        target_q_dist = (target_q_dist * a_onehot).sum(-1)  # [N, T, B, A] => [N, T, B]
        target_q_dist = target_q_dist.permute(1, 2, 0)  # [N, T, B] => [T, B, N]
        target_q_dist = target_q_dist.view(-1, self.atoms)  # [T, B, N] => [T*B, N]

        target = q_target_func(BATCH.reward.repeat(1, 1, self.atoms),
                               self.gamma,
                               BATCH.done.repeat(1, 1, self.atoms),
                               self.z.view(1, 1, self.atoms).repeat(time_step, batch_size, 1),
                               BATCH.begin_mask,
                               use_rnn=self.use_rnn)    # [T, B, N]
        target = target.clamp(self.v_min, self.v_max)  # [T, B, N]
        target = target.view(-1, self.atoms)  # [T, B, N] => [T*B, N]
        b = (target - self.v_min) / self.delta_z  # [T*B, N]
        u, l = b.ceil(), b.floor()  # [T*B, N]
        u_id, l_id = u.long(), l.long()  # [T*B, N]
        u_minus_b, b_minus_l = u - b, b - l  # [T*B, N]

        index_help = indexes.repeat(1, self.atoms)  # [T*B, 1] => [T*B, N]
        index_help = index_help.unsqueeze(-1)  # [T*B, N, 1]
        u_id = t.cat([index_help, u_id.unsqueeze(-1)], -1)    # [T*B, N, 2]
        l_id = t.cat([index_help, l_id.unsqueeze(-1)], -1)    # [T*B, N, 2]
        u_id = u_id.long().permute(2, 0, 1)  # [2, T*B, N]
        l_id = l_id.long().permute(2, 0, 1)  # [2, T*B, N]
        _cross_entropy = (target_q_dist * u_minus_b).detach() * q_dist[list(l_id)].log()\
            + (target_q_dist * b_minus_l).detach() * q_dist[list(u_id)].log()  # [T*B, N]
        td_error = cross_entropy = -_cross_entropy.sum(-1).view(time_step, batch_size)  # [T, B]

        loss = (cross_entropy*BATCH.get('isw', 1.0)).mean()   # 1

        self.oplr.step(loss)
        return td_error, dict([
            ['LEARNING_RATE/lr', self.oplr.lr],
            ['LOSS/loss', loss],
            ['Statistics/q_max', q_eval.max()],
            ['Statistics/q_min', q_eval.min()],
            ['Statistics/q_mean', q_eval.mean()]
        ])

    def _after_train(self):
        super()._after_train()
        if self.cur_train_step % self.assign_interval == 0:
            self.q_net.sync()
