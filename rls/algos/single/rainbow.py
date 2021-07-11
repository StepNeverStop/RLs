#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from rls.algos.base.off_policy import Off_Policy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import sync_params_pairs
from rls.nn.models import RainbowDueling
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


class RAINBOW(Off_Policy):
    '''
    Rainbow DQN:    https://arxiv.org/abs/1710.02298
        1. Double
        2. Dueling
        3. PrioritizedExperienceReplay
        4. N-Step
        5. Distributional
        6. Noisy Net
    '''

    def __init__(self,
                 envspec,

                 v_min=-10,
                 v_max=10,
                 atoms=51,
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
        assert not envspec.is_continuous, 'rainbow only support discrete action space'
        super().__init__(envspec=envspec, **kwargs)
        self.v_min = v_min
        self.v_max = v_max
        self.atoms = atoms
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self.z = t.tensor([self.v_min + i * self.delta_z for i in range(self.atoms)]).float().view(-1, self.atoms)  # [1, N]
        self.zb = self.z.repeat(self.a_dim, 1)  # [A, N]
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval

        self.rainbow_net = RainbowDueling(self.rep_net.h_dim,
                                          action_dim=self.a_dim,
                                          atoms=self.atoms,
                                          network_settings=network_settings)
        self.rainbow_target_net = deepcopy(self.rainbow_net)
        self.rainbow_target_net.eval()

        self._target_rep_net = deepcopy(self.rep_net)
        self._target_rep_net.eval()

        self._pairs = [(self.rainbow_target_net, self.rainbow_net),
                       (self._target_rep_net, self.rep_net)]
        sync_params_pairs(self._pairs)

        self.oplr = OPLR([self.rainbow_net, self.rep_net], lr)

        self._worker_modules.update(rep_net=self.rep_net,
                                    model=self.rainbow_net)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(oplr=self.oplr)
        self.initialize_data_buffer()

    def __call__(self, obs, evaluation=False):
        if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
            a = np.random.randint(0, self.a_dim, self.n_copys)
        else:
            a = self._get_action(obs)
        return a

    @iTensor_oNumpy
    def _get_action(self, obs):
        feat, self.cell_state = self.rep_net(obs, cell_state=self.cell_state)
        q_values = self.rainbow_net(feat)
        q = (self.zb * q_values).sum(-1)  # [B, A, N] => [B, A]
        return q.argmax(-1)  # [B, 1]

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
        batch_size = BATCH.action.shape[0]
        indexes = t.arange(batch_size).view(-1, 1)  # [B, 1]
        q_dist = self.rainbow_net(feat)  # [B, A, N]
        q_dist = (q_dist.permute(2, 0, 1) * BATCH.action).sum(-1).T  # [B, N]
        q_eval = (q_dist * self.z).sum(-1)
        target_q = self.rainbow_net(feat__)
        target_q = (self.zb * target_q).sum(-1)  # [B, A, N] => [B, A]
        a_ = target_q.argmax(-1).view(-1, 1)  # [B, 1]

        target_q_dist = self.rainbow_target_net(feat_)  # [B, A, N]
        target_q_dist = target_q_dist[list(t.cat([indexes, a_], -1).long().T)]   # [B, N]
        target = BATCH.reward.repeat(1, self.atoms) \
            + self.gamma * self.z * (1.0 - BATCH.done.repeat(1, self.atoms))  # [B, N], [1, N]* [B, N] = [B, N]
        target = target.clamp(self.v_min, self.v_max)  # [B, N]
        b = (target - self.v_min) / self.delta_z  # [B, N]
        u, l = b.ceil(), b.floor()  # [B, N]
        u_id, l_id = u.long(), l.long()  # [B, N]
        u_minus_b, b_minus_l = u - b, b - l  # [B, N]
        index_help = indexes.repeat(1, self.atoms)  # [B, N]
        index_help = index_help.unsqueeze(-1)  # [B, N, 1]
        u_id = t.cat([index_help, u_id.unsqueeze(-1)], -1)    # [B, N, 2]
        l_id = t.cat([index_help, l_id.unsqueeze(-1)], -1)    # [B, N, 2]
        _cross_entropy = (target_q_dist * u_minus_b).detach() * q_dist[list(l_id.long().T)].log() \
            + (target_q_dist * b_minus_l).detach() * q_dist[list(u_id.long().T)].log()  # [B, N]
        cross_entropy = -_cross_entropy.sum(-1)  # [B,]
        loss = (cross_entropy * isw).mean()
        td_error = cross_entropy

        self.oplr.step(loss)
        self.global_step.add_(1)
        return td_error, dict([
            ['LOSS/loss', loss],
            ['Statistics/q_max', q_eval.max()],
            ['Statistics/q_min', q_eval.min()],
            ['Statistics/q_mean', q_eval.mean()]
        ])
