#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from rls.algorithms.base.marl_off_policy import MultiAgentOffPolicy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import q_target_func
from rls.nn.models import CriticDueling
from rls.common.decorator import iTensor_oNumpy
from rls.nn.mixers import Mixer_REGISTER
from rls.nn.modules.wrappers import TargetTwin
from rls.common.specs import Data
from rls.nn.utils import OPLR


class VDN(MultiAgentOffPolicy):
    '''
    Value-Decomposition Networks For Cooperative Multi-Agent Learning, http://arxiv.org/abs/1706.05296
    TODO: RNN, multi-step
    '''
    policy_mode = 'off-policy'

    def __init__(self,
                 mixer='vdn',
                 mixer_settings={},
                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=2,
                 network_settings={
                     'share': [128],
                     'v': [128],
                     'adv': [128]
                 },
                 **kwargs):
        super().__init__(**kwargs)
        assert not any(list(self.is_continuouss.values())
                       ), 'VDN only support discrete action space'
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval

        self.q_nets = {}
        for id in set(self.model_ids):
            self.q_nets[id] = TargetTwin(CriticDueling(self.obs_specs[id],
                                                       rep_net_params=self.rep_net_params,
                                                       output_shape=self.a_dims[id],
                                                       network_settings=network_settings)).to(self.device)

        if mixer == 'qmix':
            assert self.state_spec.has_vector_observation or self.state_spec.has_visual_observation
        self.mixer = TargetTwin(
            Mixer_REGISTER[mixer](n_agents=self.n_agents_percopy,
                                  state_spec=self.state_spec,
                                  rep_net_params=self.rep_net_params,
                                  **mixer_settings)
        ).to(self.device)

        self.oplr = OPLR(tuple(self.q_nets.values())+(self.mixer,), lr)
        self._trainer_modules.update(
            {f"model_{id}": self.q_nets[id] for id in set(self.model_ids)})
        self._trainer_modules.update(mixer=self.mixer,
                                     oplr=self.oplr)

    @iTensor_oNumpy  # TODO: optimization
    def select_action(self, obs):
        acts = {}
        actions = {}
        for aid, mid in zip(self.agent_ids, self.model_ids):
            if self._is_train_mode and self.expl_expt_mng.is_random(self.cur_train_step):
                action = np.random.randint(0, self.a_dims[aid], self.n_copys)
            else:
                q_values = self.q_nets[mid](obs[aid])   # [B, A]
                action = action = q_values.argmax(-1)    # [B,]
            actions[aid] = action
            acts[aid] = Data(action=action)
        return actions, acts

    @iTensor_oNumpy
    def _train(self, BATCH_DICT):
        summaries = {}
        reward = BATCH_DICT[self.agent_ids[0]].reward    # [T, B, 1]
        done = 0.
        q_evals = []
        q_target_next_maxs = []
        for aid, mid in zip(self.agent_ids, self.model_ids):
            done += BATCH_DICT[aid].done    # [T, B, 1]

            q = self.q_nets[mid](BATCH_DICT[aid].obs)   # [T, B, A]
            q_eval = (q * BATCH_DICT[aid].action).sum(-1,
                                                      keepdim=True)  # [T, B, 1]
            q_evals.append(q_eval)  # N * [T, B, 1]

            next_q = self.q_nets[mid](BATCH_DICT[aid].obs_)  # [T, B, A]
            q_target = self.q_nets[mid].t(BATCH_DICT[aid].obs_)  # [T, B, A]

            next_max_action = next_q.argmax(-1)  # [T, B]
            next_max_action_one_hot = t.nn.functional.one_hot(
                next_max_action.squeeze(), self.a_dims[aid]).float()   # [T, B, A]

            q_target_next_max = (
                q_target * next_max_action_one_hot).sum(-1, keepdim=True)  # [T, B, 1]
            q_target_next_maxs.append(q_target_next_max)    # N * [T, B, 1]
        q_eval_all = self.mixer(q_evals, BATCH_DICT['global'].obs)  # [T, B, 1]
        q_target_next_max_all = self.mixer.t(
            q_target_next_maxs, BATCH_DICT['global'].obs_)  # [T, B, 1]

        q_target_all = q_target_func(reward,
                                     self.gamma,
                                     (1. - done > 0).float(),
                                     q_target_next_max_all,
                                     BATCH_DICT['global'].begin_mask,
                                     use_rnn=True)   # [T, B, 1]
        td_error = q_target_all - q_eval_all     # [T, B, 1]
        q_loss = td_error.square().mean()   # 1
        self.oplr.step(q_loss)

        summaries['model'] = dict([
            ['LOSS/loss', q_loss],
            ['Statistics/q_max', q_eval_all.max()],
            ['Statistics/q_min', q_eval_all.min()],
            ['Statistics/q_mean', q_eval_all.mean()]
        ])
        return summaries

    def _after_train(self):
        super()._after_train()
        if self.cur_train_step % self.assign_interval == 0:
            for q_net in self.q_nets.values():
                q_net.sync()
            self.mixer.sync()
