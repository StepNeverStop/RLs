#!/usr/bin/env python3
# encoding: utf-8


import torch as th

from rls.algorithms.single.dqn import DQN
from rls.common.decorator import iton
from rls.utils.torch_utils import n_step_return


class CQL_DQN(DQN):
    """
    Conservative Q-Learning for Offline Reinforcement Learning, http://arxiv.org/abs/2006.04779
    """
    policy_mode = 'off-policy'

    def __init__(self,
                 cql_weight=0.5,
                 **kwargs):
        raise NotImplementedError(
            "Dataset and replay buffer is under implementation. CQL is not for offline training for now.")
        # if you want to test CQL, you could remove the exception above and replace 'offline' to 'off-policy' and 
        # test it in online off-policy manner. 
        super().__init__(**kwargs)
        self._cql_weight = cql_weight

    @iton
    def _train(self, BATCH):
        q = self.q_net(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
        q_next = self.q_net.t(BATCH.obs_, begin_mask=BATCH.begin_mask)  # [T, B, A]
        q_eval = (q * BATCH.action).sum(-1, keepdim=True)  # [T, B, 1]
        q_target = n_step_return(BATCH.reward,
                                 self.gamma,
                                 BATCH.done,
                                 q_next.max(-1, keepdim=True)[0],
                                 BATCH.begin_mask,
                                 nstep=self._n_step_value).detach()  # [T, B, 1]
        td_error = q_target - q_eval  # [T, B, 1]
        q_loss = (td_error.square() * BATCH.get('isw', 1.0)).mean()  # 1

        cql1_loss = (th.logsumexp(q, dim=-1, keepdim=True) - q).mean()  # 1
        loss = q_loss + self._cql_weight * cql1_loss
        self.oplr.optimize(loss)
        self._summary_collector.add('LEARNING_RATE', 'lr', self.oplr.lr)
        self._summary_collector.add('LOSS', 'q_loss', q_loss)
        self._summary_collector.add('LOSS', 'cql1_loss', cql1_loss)
        self._summary_collector.add('LOSS', 'loss', loss)
        self._summary_collector.add('Statistics', 'q', q)
        return td_error
