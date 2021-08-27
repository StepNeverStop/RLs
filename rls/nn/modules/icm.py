
import torch as t
from torch.nn import Linear, Sequential, Tanh

from rls.common.decorator import iTensor_oNumpy
from rls.nn.activations import Act_REGISTER, default_act
from rls.nn.represent_nets import RepresentationNetwork
from rls.nn.utils import OPLR


class CuriosityModel(t.nn.Module):
    '''
    Model of Intrinsic Curiosity Module (ICM).
    Curiosity-driven Exploration by Self-supervised Prediction, https://arxiv.org/abs/1705.05363
    '''

    def __init__(self,
                 obs_spec,
                 rep_net_params,
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
        self.is_continuous = is_continuous
        self.action_dim = action_dim

        self.rep_net = RepresentationNetwork(obs_spec=obs_spec,
                                             rep_net_params=rep_net_params)

        self.feat_dim = self.rep_net.h_dim

        # S, S' => A
        self.inverse_dynamic_net = Sequential(
            Linear(self.feat_dim * 2, self.feat_dim * 2),
            Act_REGISTER[default_act](),
            Linear(self.feat_dim * 2, action_dim)
        )
        if self.is_continuous:
            self.inverse_dynamic_net.add_module('tanh', Tanh())

        # S, A => S'
        self.forward_net = Sequential(
            Linear(self.feat_dim + action_dim, self.feat_dim),
            Act_REGISTER[default_act](),
            Linear(self.feat_dim, self.feat_dim)
        )

        self.oplr = OPLR(models=[self.rep_net, self.inverse_dynamic_net, self.forward_net],
                         lr=lr)

    def forward(self, BATCH):
        fs, _ = self.rep_net(BATCH.obs)  # [T, B, *]
        fs_, _ = self.rep_net(BATCH.obs_)   # [T, B, *]

        # [T, B, *] <S, A> => S'
        s_eval = self.forward_net(t.cat((fs, BATCH.action), -1))
        LF = 0.5 * (fs_ - s_eval).square().sum(-1, keepdim=True)    # [T, B, 1]
        intrinsic_reward = self.eta * LF
        loss_forward = LF.mean()    # 1

        a_eval = self.inverse_dynamic_net(t.cat((fs, fs_), -1))  # [T, B, *]
        if self.is_continuous:
            loss_inverse = 0.5 * \
                (a_eval - BATCH.action).square().sum(-1).mean()
        else:
            idx = BATCH.action.argmax(-1)  # [T, B]
            loss_inverse = t.nn.functional.cross_entropy(
                a_eval.view(-1, self.action_dim), idx.view(-1))  # 1

        loss = (1 - self.beta) * loss_inverse + self.beta * loss_forward
        self.oplr.step(loss)
        summaries = dict([
            ['LOSS/curiosity_loss', loss],
            ['LOSS/forward_loss', loss_forward],
            ['LOSS/inverse_loss', loss_inverse]
        ])
        return intrinsic_reward, summaries
