
import uuid
import torch as t

from torch.nn import (Sequential,
                      Tanh,
                      Linear)

from rls.nn.represent_nets import DefaultRepresentationNetwork
from rls.nn.activations import default_act
from rls.nn.activations import Act_REGISTER
from rls.common.decorator import iTensor_oNumpy


class CuriosityModel(t.nn.Module):
    '''
    Model of Intrinsic Curiosity Module (ICM).
    Curiosity-driven Exploration by Self-supervised Prediction, https://arxiv.org/abs/1705.05363
    '''

    def __init__(self,
                 obs_spec,
                 representation_net_params,
                 is_continuous,
                 action_dim,
                 *,
                 eta=0.2, lr=1.0e-3, beta=0.2):
        '''
        params:
            is_continuous: sepecify whether action space is continuous(True) or discrete(False)
            action_dim: dimension of action

            eta: weight of intrinsic reward
            lr: the learning rate of curiosity model
            beta: weight factor of loss between inverse_dynamic_net and forward_net
        '''
        super().__init__()
        self.eta = eta
        self.beta = beta
        self.op = t.optim.Adam(params=self.parameters(), lr=lr)
        self.is_continuous = is_continuous

        self.repre_net = DefaultRepresentationNetwork(obs_spec=obs_spec,
                                                      representation_net_params=representation_net_params)

        self.feat_dim = self.repre_net.h_dim

        # S, S' => A
        self.inverse_dynamic_net = Sequential(
            Linear(self.feat_dim * 2, self.feat_dim * 2),
            Act_REGISTER[default_act](),
            Linear(self.feat_dim * 2, action_dim)
        )
        if is_continuous:
            self.inverse_dynamic_net.add_module('tanh', Tanh())

        # S, A => S'
        self.forward_net = Sequential(
            Linear(self.feat_dim + action_dim, self.feat_dim),
            Act_REGISTER[default_act](),
            Linear(self.feat_dim, self.feat_dim)
        )

    @iTensor_oNumpy
    def forward(self, BATCH, cell_states):
        fs, _ = self.repre_net(BATCH.obs, cell_state=cell_states['obs'])
        fs_, _ = self.repre_net(BATCH.obs_, cell_state=cell_states['obs_'])

        fsa = t.cat((fs, BATCH.action), -1)            # <S, A>
        s_eval = self.forward_net(fsa)                  # <S, A> => S'
        LF = 0.5 * (fs_ - s_eval).square().sum(-1, keepdim=True)    # [B, 1]
        intrinsic_reward = self.eta * LF
        loss_forward = LF.mean()

        f = t.cat((fs, fs_), -1)
        a_eval = self.inverse_dynamic_net(f)
        if self.is_continuous:
            loss_inverse = 0.5 * (a_eval - BATCH.action).square().sum(-1).mean()
        else:
            idx = BATCH.action.argmax(-1)  # [B, ]
            loss_inverse = t.nn.functional.cross_entropy(a_eval, idx)
        loss = (1 - self.beta) * loss_inverse + self.beta * loss_forward

        self.op.zero_grad()
        loss.backward()
        self.oplr.step()
        summaries = dict([
            ['LOSS/curiosity_loss', loss],
            ['LOSS/forward_loss', loss_forward],
            ['LOSS/inverse_loss', loss_inverse]
        ])
        return intrinsic_reward, summaries
