#!/usr/bin/env python3
# encoding: utf-8

import torch as t
import torch.nn.functional as F

from rls.algorithms.multi.vdn import VDN
from rls.common.decorator import iton
from rls.nn.mixers import Mixer_REGISTER
from rls.nn.modules.wrappers import TargetTwin
from rls.utils.torch_utils import n_step_return


class QPLEX(VDN):
    '''
    QPLEX: Duplex Dueling Multi-Agent Q-Learning, http://arxiv.org/abs/2008.01062
    '''
    policy_mode = 'off-policy'

    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
        assert self.use_rnn == True, 'assert self.use_rnn == True'
        assert len(set(list(self.a_dims.values()))
                   ) == 1, 'all agents must have same action dimension.'

    def _build_mixer(self):
        assert self._mixer_type in [
            'qplex'], "assert self._mixer_type in ['qplex']"
        if self._mixer_type in ['qplex']:
            assert self._has_global_state, 'assert self._has_global_state'
        return TargetTwin(Mixer_REGISTER[self._mixer_type](n_agents=self.n_agents_percopy,
                                                           state_spec=self.state_spec,
                                                           rep_net_params=self._rep_net_params,
                                                           a_dim=list(self.a_dims.values())[0],
                                                           **self._mixer_settings)
                          ).to(self.device)

    @iton
    def _train(self, BATCH_DICT):
        summaries = {}
        reward = BATCH_DICT[self.agent_ids[0]].reward    # [T, B, 1]
        done = 0.

        q_evals = []
        q_actions = []
        q_maxs = []

        q_target_next_choose_maxs = []
        q_target_actions = []
        q_target_next_maxs = []

        for aid, mid in zip(self.agent_ids, self.model_ids):
            done += BATCH_DICT[aid].done    # [T, B, 1]

            q = self.q_nets[mid](BATCH_DICT[aid].obs,
                                 begin_mask=BATCH_DICT['global'].begin_mask)   # [T, B, A]
            q_eval = (q * BATCH_DICT[aid].action).sum(-1, keepdim=True)  # [T, B, 1]
            q_evals.append(q_eval)  # N * [T, B, 1]
            q_actions.append(BATCH_DICT[aid].action)    # N * [T, B, A]
            q_maxs.append(q.max(-1, keepdim=True)[0])   # [T, B, 1]

            q_target = self.q_nets[mid].t(BATCH_DICT[aid].obs_,
                                          begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, A]

            # use double
            next_q = self.q_nets[mid](BATCH_DICT[aid].obs_,
                                      begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, A]

            next_max_action = next_q.argmax(-1)  # [T, B]
            next_max_action_one_hot = F.one_hot(next_max_action, self.a_dims[aid]).float()   # [T, B, A]

            q_target_next_max = (q_target * next_max_action_one_hot).sum(-1, keepdim=True)  # [T, B, 1]

            q_target_next_choose_maxs.append(q_target_next_max)    # N * [T, B, 1]
            q_target_actions.append(next_max_action_one_hot)    # N * [T, B, A]
            q_target_next_maxs.append(q_target.max(-1, keepdim=True)[0])   # N * [T, B, 1]

        q_evals = t.stack(q_evals, -1)  # [T, B, 1, N]
        q_maxs = t.stack(q_maxs, -1)  # [T, B, 1, N]
        q_target_next_choose_maxs = t.stack(q_target_next_choose_maxs, -1)  # [T, B, 1, N]
        q_target_next_maxs = t.stack(q_target_next_maxs, -1)  # [T, B, 1, N]

        q_eval_tot = self.mixer(BATCH_DICT['global'].obs,
                                q_evals,
                                q_actions,
                                q_maxs,
                                begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, 1]
        q_target_next_max_tot = self.mixer.t(BATCH_DICT['global'].obs_,
                                             q_target_next_choose_maxs,
                                             q_target_actions,
                                             q_target_next_maxs,
                                             begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, 1]

        q_target_tot = n_step_return(reward,
                                     self.gamma,
                                     (done > 0.).float(),
                                     q_target_next_max_tot,
                                     BATCH_DICT['global'].begin_mask).detach()   # [T, B, 1]
        td_error = q_target_tot - q_eval_tot     # [T, B, 1]
        q_loss = td_error.square().mean()   # 1
        self.oplr.optimize(q_loss)

        summaries['model'] = dict([
            ['LOSS/q_loss', q_loss],
            ['Statistics/q_max', q_eval_tot.max()],
            ['Statistics/q_min', q_eval_tot.min()],
            ['Statistics/q_mean', q_eval_tot.mean()]
        ])
        return td_error, summaries
