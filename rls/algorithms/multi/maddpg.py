#!/usr/bin/env python3
# encoding: utf-8

from collections import defaultdict
from typing import Dict

import torch as th
import torch.distributions as td
import torch.nn.functional as F

from rls.algorithms.base.marl_off_policy import MultiAgentOffPolicy
from rls.common.data import Data
from rls.common.decorator import iton
from rls.nn.models import ActorDct, ActorDPG, MACriticQvalueOne
from rls.nn.modules.wrappers import TargetTwin
from rls.nn.noised_actions import Noise_action_REGISTER
from rls.nn.utils import OPLR
from rls.utils.torch_utils import n_step_return


class MADDPG(MultiAgentOffPolicy):
    """
    Multi-Agent Deep Deterministic Policy Gradient, https://arxiv.org/abs/1706.02275
    """
    policy_mode = 'off-policy'

    def __init__(self,
                 polyak=0.995,
                 noise_action='ou',
                 noise_params={
                     'sigma': 0.2
                 },
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 discrete_tau=1.0,
                 network_settings={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'q': [32, 32]
                 },
                 **kwargs):
        """
        TODO: Annotation
        """
        super().__init__(**kwargs)
        self.polyak = polyak
        self.discrete_tau = discrete_tau

        self.actors, self.critics = {}, {}
        for id in set(self.model_ids):
            if self.is_continuouss[id]:
                self.actors[id] = TargetTwin(ActorDPG(self.obs_specs[id],
                                                      rep_net_params=self._rep_net_params,
                                                      output_shape=self.a_dims[id],
                                                      network_settings=network_settings['actor_continuous']),
                                             self.polyak).to(self.device)
            else:
                self.actors[id] = TargetTwin(ActorDct(self.obs_specs[id],
                                                      rep_net_params=self._rep_net_params,
                                                      output_shape=self.a_dims[id],
                                                      network_settings=network_settings['actor_discrete']),
                                             self.polyak).to(self.device)
            self.critics[id] = TargetTwin(MACriticQvalueOne(list(self.obs_specs.values()),
                                                            rep_net_params=self._rep_net_params,
                                                            action_dim=sum(
                                                                self.a_dims.values()),
                                                            network_settings=network_settings['q']),
                                          self.polyak).to(self.device)
        self.actor_oplr = OPLR(list(self.actors.values()), actor_lr, **self._oplr_params)
        self.critic_oplr = OPLR(list(self.critics.values()), critic_lr, **self._oplr_params)

        # TODO: 添加动作类型判断
        self.noised_actions = {id: Noise_action_REGISTER[noise_action](**noise_params)
                               for id in set(self.model_ids) if self.is_continuouss[id]}

        self._trainer_modules.update({f"actor_{id}": self.actors[id] for id in set(self.model_ids)})
        self._trainer_modules.update({f"critic_{id}": self.critics[id] for id in set(self.model_ids)})
        self._trainer_modules.update(actor_oplr=self.actor_oplr,
                                     critic_oplr=self.critic_oplr)

    def episode_reset(self):
        super().episode_reset()
        for noised_action in self.noised_actions.values():
            noised_action.reset()

    @iton
    def select_action(self, obs: Dict):
        acts_info = {}
        actions = {}
        for aid, mid in zip(self.agent_ids, self.model_ids):
            output = self.actors[mid](obs[aid], rnncs=self.rnncs[aid])  # [B, A]
            self.rnncs_[aid] = self.actors[mid].get_rnncs()
            if self.is_continuouss[aid]:
                mu = output  # [B, A]
                pi = self.noised_actions[mid](mu)  # [B, A]
            else:
                logits = output  # [B, A]
                mu = logits.argmax(-1)  # [B,]
                cate_dist = td.Categorical(logits=logits)
                pi = cate_dist.sample()  # [B,]
            action = pi if self._is_train_mode else mu
            acts_info[aid] = Data(action=action)
            actions[aid] = action
        return actions, acts_info

    @iton
    def _train(self, BATCH_DICT):
        """
        TODO: Annotation
        """
        summaries = defaultdict(dict)
        target_actions = {}
        for aid, mid in zip(self.agent_ids, self.model_ids):
            if self.is_continuouss[aid]:
                target_actions[aid] = self.actors[mid].t(
                    BATCH_DICT[aid].obs_, begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, A]
            else:
                target_logits = self.actors[mid].t(
                    BATCH_DICT[aid].obs_, begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, A]
                target_cate_dist = td.Categorical(logits=target_logits)
                target_pi = target_cate_dist.sample()  # [T, B]
                action_target = F.one_hot(target_pi, self.a_dims[aid]).float()  # [T, B, A]
                target_actions[aid] = action_target  # [T, B, A]
        target_actions = th.cat(list(target_actions.values()), -1)  # [T, B, N*A]

        qs, q_targets = {}, {}
        for mid in self.model_ids:
            qs[mid] = self.critics[mid]([BATCH_DICT[id].obs for id in self.agent_ids],
                                        th.cat([BATCH_DICT[id].action for id in self.agent_ids], -1))  # [T, B, 1]
            q_targets[mid] = self.critics[mid].t([BATCH_DICT[id].obs_ for id in self.agent_ids],
                                                 target_actions)  # [T, B, 1]

        q_loss = {}
        td_errors = 0.
        for aid, mid in zip(self.agent_ids, self.model_ids):
            dc_r = n_step_return(BATCH_DICT[aid].reward,
                                 self.gamma,
                                 BATCH_DICT[aid].done,
                                 q_targets[mid],
                                 BATCH_DICT['global'].begin_mask).detach()  # [T, B, 1]
            td_error = dc_r - qs[mid]  # [T, B, 1]
            td_errors += td_error
            q_loss[aid] = 0.5 * td_error.square().mean()  # 1
            summaries[aid].update({
                'Statistics/q_min': qs[mid].min(),
                'Statistics/q_mean': qs[mid].mean(),
                'Statistics/q_max': qs[mid].max()
            })
        self.critic_oplr.optimize(sum(q_loss.values()))

        actor_loss = {}
        for aid, mid in zip(self.agent_ids, self.model_ids):
            if self.is_continuouss[aid]:
                mu = self.actors[mid](BATCH_DICT[aid].obs,
                                      begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, A]
            else:
                logits = self.actors[mid](BATCH_DICT[aid].obs,
                                          begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, A]
                logp_all = logits.log_softmax(-1)  # [T, B, A]
                gumbel_noise = td.Gumbel(0, 1).sample(logp_all.shape)  # [T, B, A]
                _pi = ((logp_all + gumbel_noise) / self.discrete_tau).softmax(-1)  # [T, B, A]
                _pi_true_one_hot = F.one_hot(_pi.argmax(-1), self.a_dims[aid]).float()  # [T, B, A]
                _pi_diff = (_pi_true_one_hot - _pi).detach()  # [T, B, A]
                mu = _pi_diff + _pi  # [T, B, A]

            all_actions = {id: BATCH_DICT[id].action for id in self.agent_ids}
            all_actions[aid] = mu
            q_actor = self.critics[mid](
                [BATCH_DICT[id].obs for id in self.agent_ids],
                th.cat(list(all_actions.values()), -1),
                begin_mask=BATCH_DICT['global'].begin_mask
            )  # [T, B, 1]
            actor_loss[aid] = -q_actor.mean()  # 1

        self.actor_oplr.optimize(sum(actor_loss.values()))

        for aid in self.agent_ids:
            summaries[aid].update({
                'LOSS/actor_loss': actor_loss[aid],
                'LOSS/critic_loss': q_loss[aid]
            })
        summaries['model'].update({
            'LOSS/actor_loss', sum(actor_loss.values()),
            'LOSS/critic_loss', sum(q_loss.values())
        })
        return td_errors / self.n_agents_percopy, summaries

    def _after_train(self):
        super()._after_train()
        for actor in self.actors.values():
            actor.sync()
        for critic in self.critics.values():
            critic.sync()
