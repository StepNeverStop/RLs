import numpy as np
import torch as t

from rls.nn.mlps import MLP
from rls.nn.represent_nets import RepresentationNetwork


class QMixer(t.nn.Module):
    def __init__(self,
                 n_agents,
                 state_spec,
                 rep_net_params,

                 mixing_embed_dim=8,
                 hidden_units=[8],
                 **kwargs):
        super().__init__()

        self.rep_net = RepresentationNetwork(obs_spec=state_spec,
                                             rep_net_params=rep_net_params)
        self.embed_dim = mixing_embed_dim
        self.hyper_w_1 = MLP(input_dim=self.rep_net.h_dim, hidden_units=hidden_units,
                             layer='linear', act_fn='relu', output_shape=self.embed_dim * n_agents)
        self.hyper_w_final = MLP(input_dim=self.rep_net.h_dim, hidden_units=hidden_units,
                                 layer='linear', act_fn='relu', output_shape=self.embed_dim)
        # State dependent bias for hidden layer
        self.hyper_b_1 = t.nn.Linear(self.rep_net.h_dim, self.embed_dim)
        # V(s) instead of a bias for the last layers
        self.V = MLP(input_dim=self.rep_net.h_dim, hidden_units=[self.embed_dim],
                     layer='linear', act_fn='relu', output_shape=1)

    def forward(self, q_values, state, **kwargs):
        '''
        params:
            q_values: N * [T, B, 1]
            state: [T, B, *]
        '''
        # q_values: List[Tensor(T*B, 1)]
        time_step = q_values[0].shape[0]    # T
        batch_size = q_values[0].shape[1]   # B

        # state: [T, B, *]
        state_feat, _ = self.rep_net(state, **kwargs)    # [T, B, *]
        q_values = t.stack(q_values, -1)  # [T, B, 1, N]
        # First layer
        w1 = t.abs(self.hyper_w_1(state_feat))   # [T, B, **N]
        b1 = self.hyper_b_1(state_feat)  # [T, B, *]
        w1 = w1.view(time_step, batch_size, -1, self.embed_dim)  # [T, B, N, *]
        b1 = b1.view(time_step, batch_size, 1, self.embed_dim)  # [T, B, 1, *]
        hidden = t.nn.functional.elu(q_values @ w1 + b1)  # [T, B, 1, *]
        # Second layer
        w_final = t.abs(self.hyper_w_final(state_feat))  # [T, B, *]
        w_final = w_final.view(time_step, batch_size,
                               self.embed_dim, 1)   # [T, B, *, 1]
        # State-dependent bias
        v = self.V(state_feat).view(
            time_step, batch_size, 1, 1)   # [T, B, 1, 1]
        # Compute final output
        y = hidden @ w_final + v  # [T, B, 1, 1]
        # Reshape and return
        q_tot = y.view(time_step, batch_size, 1)   # [T, B, 1]
        # q_tot = y.squeeze(-1)  # [T, B, 1]
        return q_tot
