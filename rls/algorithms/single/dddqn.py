#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch.nn.functional as F

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.common.data import Data
from rls.common.decorator import iton
from rls.nn.models import CriticDueling
from rls.nn.modules.wrappers import TargetTwin
from rls.nn.utils import OPLR
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import n_step_return


class DDDQN(SarlOffPolicy):
    """
    Dueling Double DQN, https://arxiv.org/abs/1511.06581
    """
    policy_mode = 'off-policy'

    def __init__(self,
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
        assert not self.is_continuous, 'dueling double dqn only support discrete action space'
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self._max_train_step)
        self.assign_interval = assign_interval

        self.q_net = TargetTwin(CriticDueling(self.obs_spec,
                                              rep_net_params=self._rep_net_params,
                                              output_shape=self.a_dim,
                                              network_settings=network_settings)).to(self.device)

        self.oplr = OPLR(self.q_net, lr, **self._oplr_params)
        self._trainer_modules.update(model=self.q_net,
                                     oplr=self.oplr)

    @iton
    def select_action(self, obs):
        q_values = self.q_net(obs, rnncs=self.rnncs)  # [B, A]
        self.rnncs_ = self.q_net.get_rnncs()

        if self._is_train_mode and self.expl_expt_mng.is_random(self._cur_train_step):
            actions = np.random.randint(0, self.a_dim, self.n_copies)
        else:
            actions = q_values.argmax(-1)  # [B,]
        return actions, Data(action=actions)

    @iton
    def _train(self, BATCH):
        q = self.q_net(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
        next_q = self.q_net(BATCH.obs_, begin_mask=BATCH.begin_mask)  # [T, B, A]
        q_target = self.q_net.t(BATCH.obs_, begin_mask=BATCH.begin_mask)  # [T, B, A]

        q_eval = (q * BATCH.action).sum(-1, keepdim=True)  # [T, B, 1]
        next_max_action = next_q.argmax(-1)  # [T, B]
        next_max_action_one_hot = F.one_hot(next_max_action.squeeze(), self.a_dim).float()  # [T, B, A]

        q_target_next_max = (q_target * next_max_action_one_hot).sum(-1, keepdim=True)  # [T, B, 1]
        q_target = n_step_return(BATCH.reward,
                                 self.gamma,
                                 BATCH.done,
                                 q_target_next_max,
                                 BATCH.begin_mask).detach()  # [T, B, 1]
        td_error = q_target - q_eval  # [T, B, 1]
        q_loss = (td_error.square() * BATCH.get('isw', 1.0)).mean()  # 1
        self.oplr.optimize(q_loss)

        return td_error, {
            'LEARNING_RATE/lr': self.oplr.lr,
            'LOSS/loss': q_loss,
            'Statistics/q_max': q_eval.max(),
            'Statistics/q_min': q_eval.min(),
            'Statistics/q_mean': q_eval.mean()
        }

    def _after_train(self):
        super()._after_train()
        if self._cur_train_step % self.assign_interval == 0:
            self.q_net.sync()
