#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from rls.algos.base.off_policy import Off_Policy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import (sync_params_pairs,
                                   q_target_func)
from rls.nn.models import CriticDueling
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


class DDDQN(Off_Policy):
    '''
    Dueling Double DQN, https://arxiv.org/abs/1511.06581
    '''

    def __init__(self,
                 envspec,

                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=2,
                 network_settings={
                     'share': [128],
                     'v': [128],
                     'adv': [128]
                 },
                 **kwargs):
        assert not envspec.is_continuous, 'dueling double dqn only support discrete action space'
        super().__init__(envspec=envspec, **kwargs)
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval

        self.q_net = CriticDueling(self.rep_net.hdim,
                                   output_shape=self.a_dim,
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
            feat, self.cell_state = self.rep_net(obs, cell_state=self.cell_state)
            q_values = self.q_net(feat)
            a = q_values.argmax(-1)
        return a

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
        feat, _ = self.rep_net(BATCH.obs, cell_state=cell_states['obs'])
        feat_, _ = self._target_rep_net(BATCH.obs_, cell_state=cell_states['obs_'])
        feat__, _ = self.rep_net(BATCH.obs_, cell_state=cell_states['obs_'])

        q = self.q_net(feat)
        next_q = self.q_net(feat__)
        q_target = self.q_target_net(feat_)

        q_eval = (q * BATCH.action).sum(1, keepdim=True)
        next_max_action = next_q.argmax(1)
        next_max_action_one_hot = t.nn.functional.one_hot(next_max_action.squeeze(), self.a_dim).float()

        q_target_next_max = (q_target * next_max_action_one_hot).sum(1, keepdim=True)
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
