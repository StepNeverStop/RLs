#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t
import torch.nn.functional as F

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.common.data import Data
from rls.common.decorator import iton
from rls.nn.models import RainbowDueling
from rls.nn.modules.wrappers import TargetTwin
from rls.nn.utils import OPLR, reset_noise_layer
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import n_step_return


class RAINBOW(SarlOffPolicy):
    '''
    Rainbow DQN:    https://arxiv.org/abs/1710.02298
        1. Double
        2. Dueling
        3. PrioritizedExperienceReplay
        4. N-Step
        5. Distributional (C51)
        6. Noisy Net
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
                 assign_interval=2,
                 network_settings={
                     'share': [128],
                     'v': [128],
                     'adv': [128]
                 },
                 **kwargs):
        super().__init__(**kwargs)
        assert not self.is_continuous, 'rainbow only support discrete action space'
        self._v_min = v_min
        self._v_max = v_max
        self._atoms = atoms
        self._delta_z = (self._v_max - self._v_min) / (self._atoms - 1)
        self._z = t.linspace(self._v_min, self._v_max, self._atoms).float().to(self.device)  # [N,]
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self._max_train_step)
        self.assign_interval = assign_interval
        self.rainbow_net = TargetTwin(RainbowDueling(self.obs_spec,
                                                     rep_net_params=self._rep_net_params,
                                                     action_dim=self.a_dim,
                                                     atoms=self._atoms,
                                                     network_settings=network_settings)).to(self.device)
        self.rainbow_net.target.train()  # so that NoisyLinear takes effect
        self.oplr = OPLR(self.rainbow_net, lr, **self._oplr_params)
        self._trainer_modules.update(model=self.rainbow_net,
                                     oplr=self.oplr)

    @iton
    def select_action(self, obs):
        feat = self.rainbow_net(obs, rnncs=self.rnncs)  # [B, A, N]
        self.rnncs_ = self.rainbow_net.get_rnncs()

        if self._is_train_mode and self.expl_expt_mng.is_random(self._cur_train_step):
            actions = np.random.randint(0, self.a_dim, self.n_copys)
        else:
            q = (self._z * feat).sum(-1)  # [B, A, N] * [N,] => [B, A]
            actions = q.argmax(-1)  # [B,]
        return actions, Data(action=actions)

    @iton
    def _train(self, BATCH):
        q_dist = self.rainbow_net(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A, N]
        # [T, B, A, N] * [T, B, A, 1] => [T, B, A, N] => [T, B, N]
        q_dist = (q_dist * BATCH.action.unsqueeze(-1)).sum(-2)

        q_eval = (q_dist * self._z).sum(-1)  # [T, B, N] * [N,] => [T, B]

        target_q_dist = self.rainbow_net.t(BATCH.obs_, begin_mask=BATCH.begin_mask)  # [T, B, A, N]
        # [T, B, A, N] * [1, N] => [T, B, A]
        target_q = (target_q_dist * self._z).sum(-1)
        a_ = target_q.argmax(-1)  # [T, B]
        a_onehot = F.one_hot(a_, self.a_dim).float()  # [T, B, A]
        # [T, B, A, N] * [T, B, A, 1] => [T, B, A, N] => [T, B, N]
        target_q_dist = (target_q_dist * a_onehot.unsqueeze(-1)).sum(-2)

        target = n_step_return(BATCH.reward.repeat(1, 1, self._atoms),
                               self.gamma,
                               BATCH.done.repeat(1, 1, self._atoms),
                               target_q_dist,
                               BATCH.begin_mask.repeat(1, 1, self._atoms)).detach()    # [T, B, N]
        target = target.clamp(self._v_min, self._v_max)  # [T, B, N]
        # An amazing trick for calculating the projection gracefully.
        # ref: https://github.com/ShangtongZhang/DeepRL
        target_dist = (1 - (target.unsqueeze(-1) - self._z.view(1, 1, -1, 1)).abs() / self._delta_z).clamp(0, 1) * target_q_dist.unsqueeze(-1)  # [T, B, N, 1]
        target_dist = target_dist.sum(-1)   # [T, B, N]

        _cross_entropy = -(target_dist * t.log(q_dist + t.finfo().eps)).sum(-1, keepdim=True)  # [T, B, 1]
        loss = (_cross_entropy*BATCH.get('isw', 1.0)).mean()   # 1

        self.oplr.optimize(loss)
        return _cross_entropy, dict([
            ['LEARNING_RATE/lr', self.oplr.lr],
            ['LOSS/loss', loss],
            ['Statistics/q_max', q_eval.max()],
            ['Statistics/q_min', q_eval.min()],
            ['Statistics/q_mean', q_eval.mean()]
        ])

    def _after_train(self):
        super()._after_train()
        reset_noise_layer(self.rainbow_net)
        reset_noise_layer(self.rainbow_net.target)
        if self._cur_train_step % self.assign_interval == 0:
            self.rainbow_net.sync()
