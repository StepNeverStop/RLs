#!/usr/bin/env python3
# encoding: utf-8

from copy import deepcopy

import numpy as np
import torch as t
from torch import distributions as td

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.common.decorator import iTensor_oNumpy
from rls.common.specs import Data
from rls.nn.models import CriticQvalueAll
from rls.nn.modules.wrappers import TargetTwin
from rls.nn.utils import OPLR
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import n_step_return


class MAXSQN(SarlOffPolicy):
    '''
    https://github.com/createamind/DRL/blob/master/spinup/algos/maxsqn/maxsqn.py
    '''
    policy_mode = 'off-policy'

    def __init__(self,
                 alpha=0.2,
                 beta=0.1,
                 ployak=0.995,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 use_epsilon=False,
                 q_lr=5.0e-4,
                 alpha_lr=5.0e-4,
                 auto_adaption=True,
                 network_settings=[32, 32],
                 **kwargs):
        super().__init__(**kwargs)
        assert not self.is_continuous, 'maxsqn only support discrete action space'
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self._max_train_step)
        self.use_epsilon = use_epsilon
        self.ployak = ployak
        self.auto_adaption = auto_adaption
        self.target_entropy = beta * np.log(self.a_dim)

        self.critic = TargetTwin(CriticQvalueAll(self.obs_spec,
                                                 rep_net_params=self._rep_net_params,
                                                 output_shape=self.a_dim,
                                                 network_settings=network_settings),
                                 self.ployak).to(self.device)
        self.critic2 = deepcopy(self.critic)

        self.critic_oplr = OPLR([self.critic, self.critic2], q_lr)

        if self.auto_adaption:
            self.log_alpha = t.tensor(0., requires_grad=True).to(self.device)
            self.alpha_oplr = OPLR(self.log_alpha, alpha_lr)
            self._trainer_modules.update(alpha_oplr=self.alpha_oplr)
        else:
            self.log_alpha = t.tensor(alpha).log().to(self.device)

        self._trainer_modules.update(critic=self.critic,
                                     critic2=self.critic2,
                                     log_alpha=self.log_alpha,
                                     critic_oplr=self.critic_oplr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @iTensor_oNumpy
    def select_action(self, obs):
        q = self.critic(obs, cell_state=self.cell_state)    # [B, A]
        self.next_cell_state = self.critic.get_cell_state()

        if self.use_epsilon and self._is_train_mode and self.expl_expt_mng.is_random(self._cur_train_step):
            actions = np.random.randint(0, self.a_dim, self.n_copys)
        else:
            cate_dist = td.Categorical(logits=(q / self.alpha))
            mu = q.argmax(-1)    # [B,]
            actions = pi = cate_dist.sample()   # [B,]
        return actions, Data(action=actions)

    @iTensor_oNumpy
    def _train(self, BATCH):
        q1 = self.critic(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
        q2 = self.critic2(
            BATCH.obs, begin_mask=BATCH.begin_mask)    # [T, B, A]
        q1_eval = (q1 * BATCH.action).sum(-1, keepdim=True)  # [T, B, 1]
        q2_eval = (q2 * BATCH.action).sum(-1, keepdim=True)  # [T, B, 1]

        q1_log_probs = (q1 / (self.alpha + t.finfo().eps)
                        ).log_softmax(-1)  # [T, B, A]
        q1_entropy = -(q1_log_probs.exp() * q1_log_probs).sum(-1,
                                                              keepdim=True).mean()  # 1

        q1_target = self.critic.t(
            BATCH.obs_, begin_mask=BATCH.begin_mask)   # [T, B, A]
        q2_target = self.critic2.t(
            BATCH.obs_, begin_mask=BATCH.begin_mask)  # [T, B, A]
        q1_target_max = q1_target.max(-1, keepdim=True)[0]  # [T, B, 1]
        q1_target_log_probs = (
            q1_target / (self.alpha + t.finfo().eps)).log_softmax(-1)    # [T, B, A]
        q1_target_entropy = -(q1_target_log_probs.exp() *
                              q1_target_log_probs).sum(-1, keepdim=True)   # [T, B, 1]

        q2_target_max = q2_target.max(-1, keepdim=True)[0]   # [T, B, 1]
        # q2_target_log_probs = q2_target.log_softmax(-1)
        # q2_target_log_max = q2_target_log_probs.max(1, keepdim=True)[0]

        q_target = t.minimum(q1_target_max, q2_target_max) + \
            self.alpha * q1_target_entropy  # [T, B, 1]
        dc_r = n_step_return(BATCH.reward,
                             self.gamma,
                             BATCH.done,
                             q_target,
                             BATCH.begin_mask).detach()  # [T, B, 1]
        td_error1 = q1_eval - dc_r  # [T, B, 1]
        td_error2 = q2_eval - dc_r  # [T, B, 1]
        q1_loss = (td_error1.square()*BATCH.get('isw', 1.0)).mean()   # 1
        q2_loss = (td_error2.square()*BATCH.get('isw', 1.0)).mean()   # 1
        loss = 0.5 * (q1_loss + q2_loss)
        self.critic_oplr.optimize(loss)
        summaries = dict([
            ['LEARNING_RATE/critic_lr', self.critic_oplr.lr],
            ['LOSS/loss', loss],
            ['Statistics/log_alpha', self.log_alpha],
            ['Statistics/alpha', self.alpha],
            ['Statistics/q1_entropy', q1_entropy],
            ['Statistics/q_min', t.minimum(q1, q2).mean()],
            ['Statistics/q_mean', q1.mean()],
            ['Statistics/q_max', t.maximum(q1, q2).mean()]
        ])
        if self.auto_adaption:
            alpha_loss = -(self.alpha * (self.target_entropy -
                           q1_entropy).detach()).mean()
            self.alpha_oplr.optimize(alpha_loss)
            summaries.update([
                ['LOSS/alpha_loss', alpha_loss],
                ['LEARNING_RATE/alpha_lr', self.alpha_oplr.lr]
            ])
        return (td_error1 + td_error2) / 2, summaries

    def _after_train(self):
        super()._after_train()
        self.critic.sync()
        self.critic2.sync()
