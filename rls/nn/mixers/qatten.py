import numpy as np
import torch as t

from rls.nn.mlps import MLP
from rls.nn.represent_nets import RepresentationNetwork


class QattenMixer(t.nn.Module):

    def __init__(self,
                 n_agents: int,
                 state_spec,
                 rep_net_params,

                 agent_own_state_size: bool,
                 query_hidden_units: int,
                 query_embed_dim: int,
                 key_embed_dim: int,
                 head_hidden_units: int,
                 n_attention_head: int,
                 constrant_hidden_units: int,
                 is_weighted: bool = True):
        super().__init__()

        self.n_agents = n_agents
        self.rep_net = RepresentationNetwork(obs_spec=state_spec,
                                             rep_net_params=rep_net_params)
        self.u_dim = agent_own_state_size   # TODO: implement this

        self.query_embed_dim = query_embed_dim
        self.key_embed_dim = key_embed_dim
        self.n_attention_head = n_attention_head
        self.is_weighted = is_weighted

        self.query_embedding_layers = t.nn.ModuleList()
        self.key_embedding_layers = t.nn.ModuleList()
        for i in range(self.n_attention_head):
            self.query_embedding_layers.append(MLP(input_dim=self.rep_net.h_dim, hidden_units=query_hidden_units,
                                                   layer='linear', act_fn='relu', output_shape=query_embed_dim))
            self.key_embedding_layers.append(
                t.nn.Linear(self.u_dim, self.key_embed_dim))

        self.scaled_product_value = np.sqrt(self.query_embed_dim)

        self.head_embedding_layer = MLP(input_dim=self.rep_net.h_dim, hidden_units=head_hidden_units,
                                        layer='linear', act_fn='relu', output_shape=n_attention_head)

        self.constrant_value_layer = MLP(input_dim=self.rep_net.h_dim, hidden_units=constrant_hidden_units,
                                         layer='linear', act_fn='relu', output_shape=1)

    def forward(self, q_values, state, **kwargs):
        '''
        params:
            q_values: N * [T, B, 1]
            state: [T, B, *]
        '''
        time_step = q_values[0].shape[0]    # T
        batch_size = q_values[0].shape[1]   # B

        # state: [T, B, *]
        state_feat, _ = self.rep_net(state, **kwargs)    # [T, B, *]
        q_values = t.stack(q_values, -1)  # [T, B, 1, N]

        us = self._get_us(state_feat)   # [T, B, N, *]

        q_lambda_list = []
        for i in range(self.n_attention_head):
            state_embedding = self.query_embedding_layers[i](
                state_feat)     # [T, B, *]
            u_embedding = self.key_embedding_layers[i](us)   # [T, B, N, *]

            state_embedding = state_embedding.unsqueeze(-2)  # [T, B, 1, *]
            u_embedding = u_embedding.swapaxes(-1, -2)  # [T, B, *, N]

            raw_lambda = (state_embedding @ u_embedding) / \
                self.scaled_product_value    # [T, B, 1, N]
            q_lambda = raw_lambda.softmax(dim=-1)    # [T, B, 1, N]

            q_lambda_list.append(q_lambda)   # H * [T, B, 1, N]

        q_lambda_list = t.cat(q_lambda_list, dim=-2)  # [T, B, H, N]

        q_lambda_list = q_lambda_list.swapaxes(-1, -2)  # [T, B, N, H]

        q_h = q_values @ q_lambda_list   # [T, B, 1, H]

        if self.is_weighted:
            # shape: [-1, n_attention_head, 1]
            w_h = t.abs(self.head_embedding_layer(state_feat))  # [T, B, H]
            w_h = w_h.unsqueeze(-1)  # [T, B, H, 1]

            sum_q_h = q_h @ w_h  # [T, B, 1, 1]
            sum_q_h = sum_q_h.view(time_step, batch_size, 1)   # [T, B, 1]
        else:
            sum_q_h = q_h.sum(-1)   # [T, B, 1]

        c = self.constrant_value_layer(state_feat)  # [T, B, 1]
        q_tot = sum_q_h + c  # [T, B, 1]
        return q_tot

    def _get_us(self, state_feat):
        time_step = state_feat.shape[0]    # T
        batch_size = state_feat.shape[1]   # B
        agent_own_state_size = self.u_dim
        with t.no_grad():
            us = state_feat[:, :, :agent_own_state_size*self.n_agents].view(
                time_step, batch_size, self.n_agents, agent_own_state_size)  # [T, B, N, *]
        return us
