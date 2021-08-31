#!/usr/bin/env python3
# encoding: utf-8

import torch as t

from rls.algorithms.multi.vdn import VDN
from rls.common.decorator import iTensor_oNumpy
from rls.nn.mixers import Mixer_REGISTER
from rls.nn.modules.wrappers import TargetTwin
from rls.utils.torch_utils import n_step_return


class QTRAN(VDN):
    '''
    QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning
    http://arxiv.org/abs/1905.05408
    '''
    policy_mode = 'off-policy'

    def __init__(self,
                 opt_loss=1,
                 nopt_min_loss=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        assert self.use_rnn == True, 'assert self.use_rnn == True'
        assert len(set(list(self.a_dims.values()))
                   ) == 1, 'all agents must have same action dimension.'
        self.opt_loss = opt_loss
        self.nopt_min_loss = nopt_min_loss

    def _build_mixer(self):
        assert self._mixer_type in [
            'qtran-base'], "assert self._mixer_type in ['qtran-base']"
        if self._mixer_type in ['qtran-base']:
            assert self._has_global_state, 'assert self._has_global_state'
        return TargetTwin(
            Mixer_REGISTER[self._mixer_type](n_agents=self.n_agents_percopy,
                                             state_spec=self.state_spec,
                                             rep_net_params=self._rep_net_params,
                                             a_dim=list(
                                                 self.a_dims.values())[0],
                                             **self._mixer_settings)
        ).to(self.device)

    @iTensor_oNumpy
    def _train(self, BATCH_DICT):
        summaries = {}
        reward = BATCH_DICT[self.agent_ids[0]].reward    # [T, B, 1]
        done = 0.

        q_evals = []
        q_cell_states = []
        q_actions = []
        q_maxs = []
        q_max_actions = []

        q_target_next_choose_maxs = []
        q_target_cell_states = []
        q_target_actions = []

        for aid, mid in zip(self.agent_ids, self.model_ids):
            done += BATCH_DICT[aid].done    # [T, B, 1]

            q = self.q_nets[mid](
                BATCH_DICT[aid].obs, begin_mask=BATCH_DICT['global'].begin_mask)   # [T, B, A]
            q_cell_state = self.q_nets[mid].get_cell_state()  # [T, B, *]
            q_eval = (q * BATCH_DICT[aid].action).sum(-1,
                                                      keepdim=True)  # [T, B, 1]
            q_evals.append(q_eval)  # N * [T, B, 1]
            q_cell_states.append(q_cell_state)  # N * [T, B, *]
            q_actions.append(BATCH_DICT[aid].action)    # N * [T, B, A]
            q_maxs.append(q.max(-1, keepdim=True)[0])   # [T, B, 1]
            q_max_actions.append(t.nn.functional.one_hot(
                q.argmax(-1), self.a_dims[aid]).float())  # [T, B, A]

            q_target = self.q_nets[mid].t(
                BATCH_DICT[aid].obs_, begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, A]
            # [T, B, *]
            q_target_cell_state = self.q_nets[mid].target.get_cell_state()
            if self._use_double:
                next_q = self.q_nets[mid](
                    BATCH_DICT[aid].obs_, begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, A]

                next_max_action = next_q.argmax(-1)  # [T, B]
                next_max_action_one_hot = t.nn.functional.one_hot(
                    next_max_action, self.a_dims[aid]).float()   # [T, B, A]

                q_target_next_max = (
                    q_target * next_max_action_one_hot).sum(-1, keepdim=True)  # [T, B, 1]
            else:
                next_max_action = q_target.argmax(-1)  # [T, B]
                next_max_action_one_hot = t.nn.functional.one_hot(
                    next_max_action, self.a_dims[aid]).float()   # [T, B, A]
                # [T, B, 1]
                q_target_next_max = q_target.max(-1, keepdim=True)[0]

            q_target_next_choose_maxs.append(
                q_target_next_max)    # N * [T, B, 1]
            q_target_cell_states.append(q_target_cell_state)    # N * [T, B, *]
            q_target_actions.append(next_max_action_one_hot)    # N * [T, B, A]

        joint_qs, vs = self.mixer(BATCH_DICT['global'].obs, q_cell_states, q_actions,
                                  begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, 1]
        target_joint_qs, target_vs = self.mixer.t(BATCH_DICT['global'].obs_, q_target_cell_states, q_target_actions,
                                                  begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, 1]

        q_target_tot = n_step_return(reward,
                                     self.gamma,
                                     (done > 0.).float(),
                                     target_joint_qs,
                                     BATCH_DICT['global'].begin_mask).detach()   # [T, B, 1]
        td_error = q_target_tot - joint_qs     # [T, B, 1]
        td_loss = td_error.square().mean()   # 1

        # opt loss
        max_joint_qs, _ = self.mixer(BATCH_DICT['global'].obs, q_cell_states, q_max_actions,
                                     begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, 1]
        max_actions_qvals = sum(q_maxs)  # [T, B, 1]
        opt_loss = (max_actions_qvals - max_joint_qs.detach() +
                    vs).square().mean()   # 1

        # nopt loss
        nopt_error = sum(q_evals) - joint_qs.detach() + vs  # [T, B, 1]
        nopt_error = nopt_error.clamp(max=0)    # [T, B, 1]
        nopt_loss = nopt_error.square().mean()   # 1

        loss = td_loss + self.opt_loss * opt_loss + self.nopt_min_loss * nopt_loss

        self.oplr.optimize(loss)

        summaries['model'] = dict([
            ['LOSS/q_loss', td_loss],
            ['LOSS/loss', loss],
            ['Statistics/q_max', joint_qs.max()],
            ['Statistics/q_min', joint_qs.min()],
            ['Statistics/q_mean', joint_qs.mean()]
        ])
        return td_error, summaries
