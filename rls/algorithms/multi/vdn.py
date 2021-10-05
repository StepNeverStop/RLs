#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as th
import torch.nn.functional as F

from rls.algorithms.base.marl_off_policy import MultiAgentOffPolicy
from rls.common.data import Data
from rls.common.decorator import iton
from rls.nn.mixers import Mixer_REGISTER
from rls.nn.models import CriticDueling
from rls.nn.modules.wrappers import TargetTwin
from rls.nn.utils import OPLR
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import n_step_return


class VDN(MultiAgentOffPolicy):
    """
    Value-Decomposition Networks For Cooperative Multi-Agent Learning, http://arxiv.org/abs/1706.05296
    QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning, http://arxiv.org/abs/1803.11485
    Qatten: A General Framework for Cooperative Multiagent Reinforcement Learning, http://arxiv.org/abs/2002.03939
    """
    policy_mode = 'off-policy'

    def __init__(self,
                 mixer='vdn',
                 mixer_settings={},
                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 use_double=True,
                 init2mid_annealing_step=1000,
                 assign_interval=1000,
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
                                                          max_step=self._max_train_step)
        self.assign_interval = assign_interval
        self._use_double = use_double
        self._mixer_type = mixer
        self._mixer_settings = mixer_settings

        self.q_nets = {}
        for id in set(self.model_ids):
            self.q_nets[id] = TargetTwin(CriticDueling(self.obs_specs[id],
                                                       rep_net_params=self._rep_net_params,
                                                       output_shape=self.a_dims[id],
                                                       network_settings=network_settings)).to(self.device)

        self.mixer = self._build_mixer()

        self.oplr = OPLR(tuple(self.q_nets.values()) + (self.mixer,), lr, **self._oplr_params)
        self._trainer_modules.update({f"model_{id}": self.q_nets[id] for id in set(self.model_ids)})
        self._trainer_modules.update(mixer=self.mixer,
                                     oplr=self.oplr)

    def _build_mixer(self):
        assert self._mixer_type in [
            'vdn', 'qmix', 'qatten'], "assert self._mixer_type in ['vdn', 'qmix', 'qatten']"
        if self._mixer_type in ['qmix', 'qatten']:
            assert self._has_global_state, 'assert self._has_global_state'
        return TargetTwin(Mixer_REGISTER[self._mixer_type](n_agents=self.n_agents_percopy,
                                                           state_spec=self.state_spec,
                                                           rep_net_params=self._rep_net_params,
                                                           **self._mixer_settings)
                          ).to(self.device)

    @iton  # TODO: optimization
    def select_action(self, obs):
        acts_info = {}
        actions = {}
        for aid, mid in zip(self.agent_ids, self.model_ids):
            q_values = self.q_nets[mid](obs[aid], rnncs=self.rnncs[aid])  # [B, A]
            self.rnncs_[aid] = self.q_nets[mid].get_rnncs()

            if self._is_train_mode and self.expl_expt_mng.is_random(self._cur_train_step):
                action = np.random.randint(0, self.a_dims[aid], self.n_copies)
            else:
                action = q_values.argmax(-1)  # [B,]

            actions[aid] = action
            acts_info[aid] = Data(action=action)
        return actions, acts_info

    @iton
    def _train(self, BATCH_DICT):
        reward = BATCH_DICT[self.agent_ids[0]].reward  # [T, B, 1]
        done = 0.
        q_evals = []
        q_target_next_choose_maxs = []
        for aid, mid in zip(self.agent_ids, self.model_ids):
            done += BATCH_DICT[aid].done  # [T, B, 1]

            q = self.q_nets[mid](BATCH_DICT[aid].obs,
                                 begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, A]
            q_eval = (q * BATCH_DICT[aid].action).sum(-1,
                                                      keepdim=True)  # [T, B, 1]
            q_evals.append(q_eval)  # N * [T, B, 1]

            q_target = self.q_nets[mid].t(BATCH_DICT[aid].obs_,
                                          begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, A]
            if self._use_double:
                next_q = self.q_nets[mid](BATCH_DICT[aid].obs_,
                                          begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, A]

                next_max_action = next_q.argmax(-1)  # [T, B]
                next_max_action_one_hot = F.one_hot(next_max_action, self.a_dims[aid]).float()  # [T, B, A]

                q_target_next_max = (q_target * next_max_action_one_hot).sum(-1, keepdim=True)  # [T, B, 1]
            else:
                # [T, B, 1]
                q_target_next_max = q_target.max(-1, keepdim=True)[0]

            q_target_next_choose_maxs.append(q_target_next_max)  # N * [T, B, 1]

        q_evals = th.stack(q_evals, -1)  # [T, B, 1, N]
        q_target_next_choose_maxs = th.stack(q_target_next_choose_maxs, -1)  # [T, B, 1, N]
        q_eval_tot = self.mixer(q_evals, BATCH_DICT['global'].obs,
                                begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, 1]
        q_target_next_max_tot = self.mixer.t(q_target_next_choose_maxs, BATCH_DICT['global'].obs_,
                                             begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, 1]

        q_target_tot = n_step_return(reward,
                                     self.gamma,
                                     (done > 0.).float(),
                                     q_target_next_max_tot,
                                     BATCH_DICT['global'].begin_mask).detach()  # [T, B, 1]
        td_error = q_target_tot - q_eval_tot  # [T, B, 1]
        q_loss = td_error.square().mean()  # 1
        self.oplr.optimize(q_loss)
        self._summary_collectors['model'].add('LOSS', 'q_loss', q_loss)
        self._summary_collectors['model'].add('Statistics', 'q', q_eval_tot)

        return td_error

    def _after_train(self):
        super()._after_train()
        if self._cur_train_step % self.assign_interval == 0:
            for q_net in self.q_nets.values():
                q_net.sync()
            self.mixer.sync()
