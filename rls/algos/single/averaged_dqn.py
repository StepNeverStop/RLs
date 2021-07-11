#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from typing import (Union,
                    List,
                    NoReturn)

from rls.algos.base.off_policy import Off_Policy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import (sync_params,
                                   q_target_func)
from rls.nn.models import CriticQvalueAll
from rls.nn.utils import OPLR
from rls.utils.sundry_utils import to_numpy


class AveragedDQN(Off_Policy):
    '''
    Averaged-DQN, http://arxiv.org/abs/1611.01929
    '''

    def __init__(self,
                 envspec,

                 target_k: int = 4,
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
        self.target_k = target_k
        assert self.target_k > 0, "assert self.target_k > 0"
        self.current_target_idx = 0

        self.q_net = CriticQvalueAll(self.rep_net.h_dim,
                                     output_shape=self.a_dim,
                                     network_settings=network_settings)
        self.target_representation_nets = []
        self.target_nets = []
        for i in range(self.target_k):
            target_rep_net = deepcopy(self.rep_net)
            target_rep_net.eval()
            sync_params(target_rep_net, self.rep_net)
            target_q_net = deepcopy(self.q_net)
            target_q_net.eval()
            sync_params(target_q_net, self.q_net)
            self.target_representation_nets.append(target_rep_net)
            self.target_nets.append(target_q_net)

        self.oplr = OPLR([self.q_net, self.rep_net], lr)

        self._worker_modules.update(rep_net=self.rep_net,
                                    model=self.q_net)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(oplr=self.oplr)
        self.initialize_data_buffer()

    def __call__(self, obs, evaluation: bool = False) -> np.ndarray:
        if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
            a = np.random.randint(0, self.a_dim, self.n_copys)
        else:
            a, self.cell_state = self._get_action(obs, self.cell_state)
        return a

    def _get_action(self, obs, cell_state):
        feat, cell_state = self.rep_net(obs.tensor, cell_state=cell_state)
        q_values = self.q_net(feat)
        for i in range(1, self.target_k):
            target_feat, _ = self.target_representation_nets[i](obs, cell_state=cell_state)
            target_q_values = self.target_nets[i](target_feat)
            q_values += target_q_values
        return to_numpy(q_values.argmax(1)), cell_state  # 不取平均也可以

    def _target_params_update(self):
        if self.global_step % self.assign_interval == 0:
            sync_params(self.representation_net_params[self, current_target_idx], self.rep_net)
            sync_params(self.target_nets[self.current_target_idx], self.q_net)
            self.current_target_idx = (self.current_target_idx + 1) % self.target_k

    def learn(self, **kwargs) -> NoReturn:
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([['LEARNING_RATE/lr', self.oplr.lr]])
            })

    def _train(self, BATCH, isw, cell_states):
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
        q = self.q_net(feat)
        feat_, _ = self.target_representation_nets[0](BATCH.obs_, cell_state=cell_states['obs_'])
        q_next = self.target_nets[0](feat_)
        for i in range(1, self.target_k):
            feat_, _ = self.target_representation_nets[0](BATCH.obs_, cell_state=cell_states['obs_'])
            q_next += self.target_nets[0](feat_)
        q_next /= self.target_k
        q_eval = (q * BATCH.action).sum(1, keepdim=True)
        q_target = q_target_func(BATCH.reward,
                                 self.gamma,
                                 BATCH.done,
                                 q_next.max(-1, keepdim=True)[0])
        td_error = q_target - q_eval

        self.oplr.step(q_loss)
        self.global_step.add_(1)
        return td_error, dict([
            ['LOSS/loss', q_loss],
            ['Statistics/q_max', q_eval.max()],
            ['Statistics/q_min', q_eval.min()],
            ['Statistics/q_mean', q_eval.mean()]
        ])
