import numpy as np
import torch as t
import torch.nn as nn

from rls.nn.mlps import MLP


class SI_Weight(nn.Module):
    """https://github.com/wjh720/QPLEX/"""

    def __init__(self,
                 n_agents,
                 state_feat_dim,
                 a_dim,
                 num_kernel,
                 adv_hidden_units):
        super().__init__()

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()
        for i in range(num_kernel):  # multi-head attention
            self.key_extractors.append(MLP(input_dim=state_feat_dim, hidden_units=adv_hidden_units,
                                           layer='linear', act_fn='relu', output_shape=1))  # key
            self.agents_extractors.append(MLP(input_dim=state_feat_dim, hidden_units=adv_hidden_units,
                                              layer='linear', act_fn='relu', output_shape=n_agents))  # agent
            self.action_extractors.append(MLP(input_dim=state_feat_dim + n_agents * a_dim, hidden_units=adv_hidden_units,
                                              layer='linear', act_fn='relu', output_shape=n_agents))  # action

    def forward(self, state_feat, actions):
        '''
        state_feat: [T, B, *]
        actions: N * [T, B, A]
        '''
        data = t.cat([state_feat]+actions, dim=-1)  # [T, B, *]

        all_head_key = [k_ext(state_feat)
                        for k_ext in self.key_extractors]  # List[[T, B, 1]]
        all_head_agents = [k_ext(state_feat)
                           for k_ext in self.agents_extractors]   # List[[T, B, N]]
        # List[[T, B, N]]
        all_head_action = [sel_ext(data) for sel_ext in self.action_extractors]

        head_attend_weights = []
        for curr_head_key, curr_head_agents, curr_head_action in zip(all_head_key, all_head_agents, all_head_action):
            x_key = t.abs(curr_head_key) + t.finfo().eps    # [T, B, 1]
            x_agents = curr_head_agents.sigmoid()     # [T, B, N]
            x_action = curr_head_action.sigmoid()     # [T, B, N]
            weights = x_key * x_agents * x_action   # [T, B, N]
            head_attend_weights.append(weights)  # List[[T, B, N]]

        head_attend = sum(head_attend_weights)  # [T, B, N]
        return head_attend
