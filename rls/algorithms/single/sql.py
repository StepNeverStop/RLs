#!/usr/bin/env python3
# encoding: utf-8

import torch.distributions as td

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.common.data import Data
from rls.common.decorator import iton
from rls.nn.models import CriticQvalueAll
from rls.nn.modules.wrappers import TargetTwin
from rls.nn.utils import OPLR
from rls.utils.torch_utils import n_step_return


class SQL(SarlOffPolicy):
    """
        Soft Q-Learning. ref: https://github.com/Bigpig4396/PyTorch-Soft-Q-Learning/blob/master/SoftQ.py
        NOTE: not the original of the paper, NO SVGD.
        Reinforcement Learning with Deep Energy-Based Policies: https://arxiv.org/abs/1702.08165
    """
    policy_mode = 'off-policy'

    def __init__(self,
                 lr=5.0e-4,
                 alpha=2,
                 polyak=0.995,
                 network_settings=[32, 32],
                 **kwargs):
        super().__init__(**kwargs)
        assert not self.is_continuous, 'sql only support discrete action space'
        self.alpha = alpha
        self.polyak = polyak

        self.q_net = TargetTwin(CriticQvalueAll(self.obs_spec,
                                                rep_net_params=self._rep_net_params,
                                                output_shape=self.a_dim,
                                                network_settings=network_settings),
                                self.polyak).to(self.device)

        self.oplr = OPLR(self.q_net, lr, **self._oplr_params)
        self._trainer_modules.update(model=self.q_net,
                                     oplr=self.oplr)

    @iton
    def select_action(self, obs):
        q_values = self.q_net(obs, rnncs=self.rnncs)  # [B, A]
        self.rnncs_ = self.q_net.get_rnncs()
        logits = ((q_values - self._get_v(q_values)) / self.alpha).exp()  # > 0   # [B, A]
        logits /= logits.sum(-1, keepdim=True)  # [B, A]
        cate_dist = td.Categorical(logits=logits)
        actions = cate_dist.sample()  # [B,]
        return actions, Data(action=actions)

    def _get_v(self, q):
        v = self.alpha * (q / self.alpha).exp().mean(-1, keepdim=True).log()  # [B, 1] or [T, B, 1]
        return v

    @iton
    def _train(self, BATCH):
        q = self.q_net(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, A]
        q_next = self.q_net.t(BATCH.obs_, begin_mask=BATCH.begin_mask)  # [T, B, A]
        v_next = self._get_v(q_next)  # [T, B, 1]
        q_eval = (q * BATCH.action).sum(-1, keepdim=True)  # [T, B, 1]
        q_target = n_step_return(BATCH.reward,
                                 self.gamma,
                                 BATCH.done,
                                 v_next,
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
        self.q_net.sync()
