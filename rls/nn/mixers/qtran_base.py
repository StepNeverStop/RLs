from typing import Dict, List

import torch as th
import torch.nn as nn

from rls.nn.mlps import MLP
from rls.nn.represent_nets import RepresentationNetwork


class QTranBase(nn.Module):

    def __init__(self,
                 n_agents,
                 state_spec,
                 rep_net_params,
                 a_dim,

                 qtran_arch,
                 hidden_units):
        super().__init__()

        self.rep_net = RepresentationNetwork(obs_spec=state_spec,
                                             rep_net_params=rep_net_params)
        self.qtran_arch = qtran_arch  # QTran architecture

        self.h_nums = 2 if self.rep_net.memory_net.network_type == 'lstm' else 1

        # Q takes [state, agent_action_observation_encodings]

        # Q(s,u)
        if self.qtran_arch == "coma_critic":
            # Q takes [state, u] as input
            q_input_size = self.rep_net.h_dim + (n_agents * a_dim)
        elif self.qtran_arch == "qtran_paper":
            # Q takes [state, agent_action_observation_encodings]
            ae_input = self.h_nums * self.rep_net.h_dim + a_dim
            self.action_encoding = MLP(input_dim=ae_input, hidden_units=[ae_input],
                                       layer='linear', act_fn='relu', output_shape=ae_input)
            q_input_size = self.rep_net.h_dim + ae_input
        else:
            raise Exception(
                "{} is not a valid QTran architecture".format(self.qtran_arch))

        self.Q = MLP(input_dim=q_input_size, hidden_units=hidden_units,
                     layer='linear', act_fn='relu', output_shape=1)
        # V(s)
        self.V = MLP(input_dim=self.rep_net.h_dim, hidden_units=hidden_units,
                     layer='linear', act_fn='relu', output_shape=1)

    def forward(self, state, hidden_states: List[Dict[str, th.Tensor]], actions: List[th.Tensor], **kwargs):
        """
        state: [T, B, *]
        hidden_states: N * [T, B, *]
        actions: N * [T, B, A]
        """

        # state: [T, B, *]
        state_feat, _ = self.rep_net(state, **kwargs)  # [T, B, *]

        if self.qtran_arch == "coma_critic":
            actions = th.cat(actions, dim=-1)  # [T, B, N*A]
            inputs = th.cat([state_feat, actions], dim=-1)  # [T, B, *]
        elif self.qtran_arch == "qtran_paper":
            hs = [th.cat(list(hidden_state.values()), -1)
                  for hidden_state in hidden_states]  # N * [T, B, *]
            hs = th.stack(hs, dim=-2)  # [T, B, N, *]
            actions = th.stack(actions, dim=-2)  # [T, B, N, A]
            _input = th.cat((hs, actions), dim=-1)  # [T, B, N, *]

            agent_state_action_encoding = self.action_encoding(
                _input)  # [T, B, N, *]
            agent_state_action_encoding = agent_state_action_encoding.sum(
                dim=-2)  # [T, B, *]

            inputs = th.cat(
                [state_feat, agent_state_action_encoding], dim=-1)  # [T, B, *]

        q_outputs = self.Q(inputs)  # [T, B, 1]
        v_outputs = self.V(state_feat)  # [T, B, 1]
        return q_outputs, v_outputs
