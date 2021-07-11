#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from rls.algos.base.off_policy import Off_Policy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import (huber_loss,
                                   q_target_func,
                                   sync_params)
from rls.nn.models import IqnNet
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


class IQN(Off_Policy):
    '''
    Implicit Quantile Networks, https://arxiv.org/abs/1806.06923
    Double DQN
    '''

    def __init__(self,
                 envspec,

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
        assert not envspec.is_continuous, 'iqn only support discrete action space'
        super().__init__(envspec=envspec, **kwargs)
        self.pi = t.tensor(np.pi)
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

        self.q_net = IqnNet(self.rep_net.h_dim,
                            action_dim=self.a_dim,
                            quantiles_idx=self.quantiles_idx,
                            network_settings=network_settings)
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

    @iTensor_oNumpy
    def __call__(self, obs, evaluation=False):
        if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
            a = np.random.randint(0, self.a_dim, self.n_copys)
        else:
            batch_size = obs.shape[0]   # TODO
            _, select_quantiles_tiled = self._generate_quantiles(   # [N*B, 64]
                batch_size=batch_size,
                quantiles_num=self.select_quantiles,
                quantiles_idx=self.quantiles_idx
            )
            # [B, A]
            feat, self.cell_state = self.rep_net(obs, cell_state=self.cell_state)
            (_, q_values) = self.q_net(feat, elect_quantiles_tiled, quantiles_num=self.select_quantiles)
            a = q_values.argmax(-1)  # [B,]
        return a

    def _generate_quantiles(self, batch_size, quantiles_num, quantiles_idx):
        _quantiles = t.rand([batch_size * quantiles_num, 1])  # [N*B, 1]
        _quantiles_tiled = _quantiles.repeat(1, quantiles_idx)  # [N*B, 1] => [N*B, 64]
        _quantiles_tiled = t.arange(quantiles_idx) * self.pi * _quantiles_tiled  # pi * i * tau [N*B, 64] * [64, ] => [N*B, 64]
        _quantiles_tiled.cos_()   # [N*B, 64]
        _quantiles = _quantiles.view(batch_size, quantiles_num, 1)    # [N*B, 1] => [B, N, 1]
        return _quantiles, _quantiles_tiled

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

        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_state['obs'])
        feat_, _ = self._target_rep_net(BATCH.obs_, cell_state=cell_state['obs_'])
        feat__, _ = self.rep_net(BATCH.obs_, cell_state=cell_states['obs_'])

        batch_size = BATCH.action.shape[0]
        quantiles, quantiles_tiled = self._generate_quantiles(   # [B, N, 1], [N*B, 64]
            batch_size=batch_size,
            quantiles_num=self.online_quantiles,
            quantiles_idx=self.quantiles_idx
        )
        quantiles_value, q = self.q_net(feat, quantiles_tiled, quantiles_num=self.online_quantiles)    # [N, B, A], [B, A]
        _a = BATCH.action.repeat(self.online_quantiles, 1).view(self.online_quantiles, -1, self.a_dim)  # [B, A] => [N*B, A] => [N, B, A]
        quantiles_value = (quantiles_value * _a).sum(-1, keepdim=True)   # [N, B, A] => [N, B, 1]
        q_eval = (q * BATCH.action).sum(-1, keepdim=True)  # [B, A] => [B, 1]

        _, select_quantiles_tiled = self._generate_quantiles(   # [N*B, 64]
            batch_size=batch_size,
            quantiles_num=self.select_quantiles,
            quantiles_idx=self.quantiles_idx
        )
        _, q_values = self.q_net(feat__, select_quantiles_tiled, quantiles_num=self.select_quantiles)  # [B, A]
        next_max_action = q_values.argmax(-1)   # [B,]
        next_max_action = t.nn.functional.one_hot(next_max_action.squeeze(), self.a_dim).float()  # [B, A]
        _next_max_action = next_max_action.repeat(self.target_quantiles, 1).view(self.target_quantiles, -1, self.a_dim)  # [B, A] => [N'*B, A] => [N', B, A]
        _, target_quantiles_tiled = self._generate_quantiles(   # [N'*B, 64]
            batch_size=batch_size,
            quantiles_num=self.target_quantiles,
            quantiles_idx=self.quantiles_idx
        )

        target_quantiles_value, target_q = self.q_target_net(feat_, target_quantiles_tiled, quantiles_num=self.target_quantiles)  # [N', B, A], [B, A]
        target_quantiles_value = (arget_quantiles_value * _next_max_action).sum(-1, keepdim=True)   # [N', B, A] => [N', B, 1]
        target_q = (target_q * BATCH.action).sum(-1, keepdim=True)  # [B, A] => [B, 1]
        q_target = q_target_func(BATCH.reward,
                                 self.gamma,
                                 BATCH.done,
                                 target_q)   # [B, 1]
        td_error = q_target - q_eval    # [B, 1]

        _r = BATCH.reward.repeat(self.target_quantiles, 1).view(self.target_quantiles, -1, 1)  # [B, 1] => [N'*B, 1] => [N', B, 1]
        _done = BATCH.done.repeat(self.target_quantiles, 1).view(self.target_quantiles, -1, 1)    # [B, 1] => [N'*B, 1] => [N', B, 1]

        quantiles_value_target = q_target_func(_r,
                                               self.gamma,
                                               _done,
                                               target_quantiles_value)  # [N', B, 1]
        quantiles_value_target = quantiles_value_target.permute(1, 2, 0)    # [B, 1, N']
        quantiles_value_online = quantiles_value.permute(1, 0, 2)   # [B, N, 1]
        quantile_error = quantiles_value_online - quantiles_value_target    # [B, N, 1] - [B, 1, N'] => [B, N, N']
        huber = huber_loss(quantile_error, delta=self.huber_delta)  # [B, N, N']
        huber_abs = (quantiles - t.where(quantile_error < 0, 1., 0.)).abs()   # [B, N, 1] - [B, N, N'] => [B, N, N']
        loss = (huber_abs * huber).mean(-1)  # [B, N, N'] => [B, N]
        loss = loss.sum(-1)  # [B, N] => [B, ]
        loss = (loss * isw).mean()  # [B, ] => 1
        self.oplr.step(loss)
        self.global_step.add_(1)
        return td_error, dict([
            ['LOSS/loss', loss],
            ['Statistics/q_max', q_eval.max()],
            ['Statistics/q_min', q_eval.min()],
            ['Statistics/q_mean', q_eval.mean()]
        ])
