#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import q_target_func
from rls.nn.models import IqnNet
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy
from rls.nn.modules.wrappers import TargetTwin
from rls.common.specs import Data


class IQN(SarlOffPolicy):
    '''
    Implicit Quantile Networks, https://arxiv.org/abs/1806.06923
    Double DQN
    '''
    policy_mode = 'off-policy'

    def __init__(self,
                 online_quantiles=8,
                 target_quantiles=8,
                 select_quantiles=32,
                 quantiles_idx=64,
                 huber_delta=1.,
                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=2,
                 network_settings={
                     'q_net': [128, 64],
                     'quantile': [128, 64],
                     'tile': [64]
                 },
                 **kwargs):
        super().__init__(**kwargs)
        assert not self.is_continuous, 'iqn only support discrete action space'
        self.online_quantiles = online_quantiles
        self.target_quantiles = target_quantiles
        self.select_quantiles = select_quantiles
        self.quantiles_idx = quantiles_idx
        self.huber_delta = huber_delta
        self.assign_interval = assign_interval
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.q_net = TargetTwin(IqnNet(self.obs_spec,
                                       rep_net_params=self._rep_net_params,
                                       action_dim=self.a_dim,
                                       quantiles_idx=self.quantiles_idx,
                                       network_settings=network_settings)).to(self.device)
        self.oplr = OPLR(self.q_net, lr)
        self._trainer_modules.update(model=self.q_net,
                                     oplr=self.oplr)

    @iTensor_oNumpy
    def select_action(self, obs):
        if self._is_train_mode and self.expl_expt_mng.is_random(self.cur_train_step):
            actions = np.random.randint(0, self.a_dim, self.n_copys)
        else:
            _, select_quantiles_tiled = self._generate_quantiles(   # [N*B, X]
                batch_size=self.n_copys,
                quantiles_num=self.select_quantiles
            )
            q_values = self.q_net(
                obs, select_quantiles_tiled, cell_state=self.cell_state)  # [N, B, A]
            self.next_cell_state = self.q_net.get_cell_state()
            # [N, B, A] => [B, A] => [B,]
            actions = q_values.mean(0).argmax(-1)
        return actions, Data(action=actions)

    def _generate_quantiles(self, batch_size, quantiles_num):
        _quantiles = t.rand([quantiles_num*batch_size, 1])  # [N*B, 1]
        _quantiles_tiled = _quantiles.repeat(
            1, self.quantiles_idx)  # [N*B, 1] => [N*B, X]

        # pi * i * tau [N*B, X] * [X, ] => [N*B, X]
        _quantiles_tiled = t.arange(
            self.quantiles_idx) * np.pi * _quantiles_tiled
        _quantiles_tiled.cos_()   # [N*B, X]

        _quantiles = _quantiles.view(
            batch_size, quantiles_num, 1)    # [N*B, 1] => [B, N, 1]
        return _quantiles, _quantiles_tiled  # [B, N, 1], [N*B, X]

    @iTensor_oNumpy
    def _train(self, BATCH):
        time_step = BATCH.reward.shape[0]
        batch_size = BATCH.reward.shape[1]

        quantiles, quantiles_tiled = self._generate_quantiles(   # [T*B, N, 1], [N*T*B, X]
            batch_size=time_step*batch_size,
            quantiles_num=self.online_quantiles)
        # [T*B, N, 1] => [T, B, N, 1]
        quantiles = quantiles.view(time_step, batch_size, -1, 1)
        quantiles_tiled = quantiles_tiled.view(
            time_step, -1, self.quantiles_idx)    # [N*T*B, X] => [T, N*B, X]

        quantiles_value = self.q_net(
            BATCH.obs, quantiles_tiled)    # [T, N, B, A]
        # [T, N, B, A] => [N, T, B, A] * [T, B, A] => [N, T, B, 1]
        quantiles_value = (quantiles_value.swapaxes(
            0, 1) * BATCH.action).sum(-1, keepdim=True)
        q_eval = quantiles_value.mean(0)  # [N, T, B, 1] => [T, B, 1]

        _, select_quantiles_tiled = self._generate_quantiles(   # [N*T*B, X]
            batch_size=time_step*batch_size,
            quantiles_num=self.select_quantiles)
        select_quantiles_tiled = select_quantiles_tiled.view(
            time_step, -1, self.quantiles_idx)  # [N*T*B, X] => [T, N*B, X]

        q_values = self.q_net(
            BATCH.obs_, select_quantiles_tiled)  # [T, N, B, A]
        q_values = q_values.mean(1)  # [T, N, B, A] => [T, B, A]
        next_max_action = q_values.argmax(-1)   # [T, B]
        next_max_action = t.nn.functional.one_hot(
            next_max_action, self.a_dim).float()  # [T, B, A]

        _, target_quantiles_tiled = self._generate_quantiles(   # [N'*T*B, X]
            batch_size=time_step*batch_size,
            quantiles_num=self.target_quantiles)
        target_quantiles_tiled = target_quantiles_tiled.view(
            time_step, -1, self.quantiles_idx)  # [N'*T*B, X] => [T, N'*B, X]
        target_quantiles_value = self.q_net.t(
            BATCH.obs_, target_quantiles_tiled)  # [T, N', B, A]
        target_quantiles_value = target_quantiles_value.swapaxes(
            0, 1)  # [T, N', B, A] => [N', T, B, A]
        target_quantiles_value = (
            target_quantiles_value * next_max_action).sum(-1, keepdim=True)   # [N', T, B, 1]

        target_q = target_quantiles_value.mean(0)  # [T, B, 1]
        q_target = q_target_func(BATCH.reward,  # [T, B, 1]
                                 self.gamma,
                                 BATCH.done,    # [T, B, 1]
                                 target_q,  # [T, B, 1]
                                 BATCH.begin_mask,  # [T, B, 1]
                                 use_rnn=self.use_rnn)   # [T, B, 1]
        td_error = q_target - q_eval    # [T, B, 1]

        # [N', T, B, 1] => [N', T, B]
        target_quantiles_value = target_quantiles_value.squeeze(-1)
        target_quantiles_value = target_quantiles_value.permute(
            1, 2, 0)    # [N', T, B] => [T, B, N']
        quantiles_value_target = q_target_func(BATCH.reward.repeat(1, 1, self.target_quantiles),
                                               self.gamma,
                                               BATCH.done.repeat(
                                                   1, 1, self.target_quantiles),
                                               target_quantiles_value,
                                               BATCH.begin_mask.repeat(
                                                   1, 1, self.target_quantiles),
                                               use_rnn=self.use_rnn)  # [T, B, N']
        # [T, B, N'] => [T, B, 1, N']
        quantiles_value_target = quantiles_value_target.unsqueeze(-2)
        quantiles_value_online = quantiles_value.permute(
            1, 2, 0, 3)   # [N, T, B, 1] => [T, B, N, 1]
        # [T, B, N, 1] - [T, B, 1, N'] => [T, B, N, N']
        quantile_error = quantiles_value_online - quantiles_value_target
        huber = t.nn.functional.huber_loss(
            quantiles_value_online, quantiles_value_target, reduction="none", delta=self.huber_delta)    # [T, B, N, N]
        # [T, B, N, 1] - [T, B, N, N'] => [T, B, N, N']
        huber_abs = (quantiles - quantile_error.detach().le(0.).float()).abs()
        loss = (huber_abs * huber).mean(-1)  # [T, B, N, N'] => [T, B, N]
        loss = loss.sum(-1, keepdim=True)  # [T, B, N] => [T, B, 1]

        loss = (loss*BATCH.get('isw', 1.0)).mean()   # 1
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
