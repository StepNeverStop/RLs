import torch as th
import torch.nn as nn

from rls.nn.mlps import MLP
from rls.nn.represent_nets import RepresentationNetwork
from .si_weight import SI_Weight


class QPLEXMixer(nn.Module):
    """https://github.com/wjh720/QPLEX/"""

    def __init__(self,
                 n_agents,
                 a_dim,
                 state_spec,
                 rep_net_params,

                 hidden_units,
                 is_minus_one,
                 weighted_head,

                 num_kernel,
                 adv_hidden_units):
        super().__init__()

        self.rep_net = RepresentationNetwork(obs_spec=state_spec,
                                             rep_net_params=rep_net_params)

        self.is_minus_one = is_minus_one
        self.weighted_head = weighted_head

        self.hyper_w_final = MLP(input_dim=self.rep_net.h_dim, hidden_units=hidden_units,
                                 layer='linear', act_fn='relu', output_shape=n_agents)
        self.V = MLP(input_dim=self.rep_net.h_dim, hidden_units=hidden_units,
                     layer='linear', act_fn='relu', output_shape=n_agents)

        self.si_weight = SI_Weight(n_agents=n_agents,
                                   state_feat_dim=self.rep_net.h_dim,
                                   a_dim=a_dim,
                                   num_kernel=num_kernel,
                                   adv_hidden_units=adv_hidden_units)

    def forward(self, state, q_values, actions, max_q_i, **kwargs):
        """
        state: [T, B, *]
        q_values: [T, B, 1, N]
        actions: N * [T, B, A]
        max_q_i: [T, B, 1, N]
        """

        time_step = q_values.shape[0]  # T
        batch_size = q_values.shape[1]  # B

        q_values = q_values.squeeze(-2)  # [T, B, N]
        max_q_i = max_q_i.squeeze(-2)  # [T, B, N]

        # state: [T, B, *]
        state_feat, _ = self.rep_net(state, **kwargs)  # [T, B, *]
        w_final = self.hyper_w_final(state_feat)  # [T, B, N]
        w_final = th.abs(w_final) + th.finfo().eps  # [T, B, N]

        v = self.V(state_feat)  # [T, B, N]

        if self.weighted_head:
            q_values = w_final * q_values + v  # [T, B, N]
            max_q_i = w_final * max_q_i + v  # [T, B, N]

        v_tot = q_values.sum(-1, keepdim=True)  # [T, B, 1]

        # adv
        adv_q = (q_values - max_q_i).detach()  # [T, B, N]

        adv_w_final = self.si_weight(state_feat, actions)  # [T, B, N]

        if self.is_minus_one:
            adv_tot = th.sum(adv_q * (adv_w_final - 1.),
                             dim=-1, keepdim=True)  # [T, B, 1]
        else:
            adv_tot = th.sum(adv_q * adv_w_final, dim=-
            1, keepdim=True)  # [T, B, 1]

        q_tot = v_tot + adv_tot  # [T, B, 1]

        return q_tot
