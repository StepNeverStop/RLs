#!/usr/bin/env python3
# encoding: utf-8

from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, NoReturn, Union

import numpy as np
import torch as t
import torch.nn.functional as F
from torch import distributions as td

from rls.algorithms.base.marl_off_policy import MultiAgentOffPolicy
from rls.common.data import Data
from rls.common.decorator import iton
from rls.nn.models import ActorCts, ActorDct, MACriticQvalueOne
from rls.nn.modules.wrappers import TargetTwin
from rls.nn.utils import OPLR
from rls.utils.sundry_utils import LinearAnnealing
from rls.utils.torch_utils import n_step_return, squash_action


class MASAC(MultiAgentOffPolicy):
    policy_mode = 'off-policy'

    def __init__(self,
                 alpha=0.2,
                 annealing=True,
                 last_alpha=0.01,
                 ployak=0.995,
                 discrete_tau=1.0,
                 network_settings={
                     'actor_continuous': {
                         'share': [128, 128],
                         'mu': [64],
                         'log_std': [64],
                         'soft_clip': False,
                         'log_std_bound': [-20, 2]
                     },
                     'actor_discrete': [64, 32],
                     'q': [128, 128]
                 },
                 auto_adaption=True,
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 alpha_lr=5.0e-4,
                 **kwargs):
        '''
        TODO: Annotation
        '''
        super().__init__(**kwargs)
        self.ployak = ployak
        self.discrete_tau = discrete_tau
        self.auto_adaption = auto_adaption
        self.annealing = annealing

        self.target_entropy = 0.98
        for id in self.agent_ids:
            if self.is_continuouss[id]:
                self.target_entropy *= (-self.a_dims[id])
            else:
                self.target_entropy *= np.log(self.a_dims[id])

        self.actors, self.critics, self.critics2 = {}, {}, {}
        for id in set(self.model_ids):
            if self.is_continuouss[id]:
                self.actors[id] = ActorCts(self.obs_specs[id],
                                           rep_net_params=self._rep_net_params,
                                           output_shape=self.a_dims[id],
                                           network_settings=network_settings['actor_continuous']).to(self.device)
            else:
                self.actors[id] = ActorDct(self.obs_specs[id],
                                           rep_net_params=self._rep_net_params,
                                           output_shape=self.a_dims[id],
                                           network_settings=network_settings['actor_discrete']).to(self.device)
            self.critics[id] = TargetTwin(MACriticQvalueOne(list(self.obs_specs.values()),
                                                            rep_net_params=self._rep_net_params,
                                                            action_dim=sum(self.a_dims.values()),
                                                            network_settings=network_settings['q']),
                                          self.ployak).to(self.device)
            self.critics2[id] = deepcopy(self.critics[id])
        self.actor_oplr = OPLR(list(self.actors.values()), actor_lr, **self._oplr_params)
        self.critic_oplr = OPLR(list(self.critics.values())+list(self.critics2.values()), critic_lr, **self._oplr_params)

        if self.auto_adaption:
            self.log_alpha = t.tensor(0., requires_grad=True).to(self.device)
            self.alpha_oplr = OPLR(self.log_alpha, alpha_lr, **self._oplr_params)
            self._trainer_modules.update(alpha_oplr=self.alpha_oplr)
        else:
            self.log_alpha = t.tensor(alpha).log().to(self.device)
            if self.annealing:
                self.alpha_annealing = LinearAnnealing(alpha, last_alpha, 1e6)

        self._trainer_modules.update({f"actor_{id}": self.actors[id] for id in set(self.model_ids)})
        self._trainer_modules.update({f"critic_{id}": self.critics[id] for id in set(self.model_ids)})
        self._trainer_modules.update({f"critic2_{id}": self.critics2[id] for id in set(self.model_ids)})
        self._trainer_modules.update(log_alpha=self.log_alpha,
                                     actor_oplr=self.actor_oplr,
                                     critic_oplr=self.critic_oplr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @iton
    def select_action(self, obs: Dict):
        acts_info = {}
        actions = {}
        for aid, mid in zip(self.agent_ids, self.model_ids):
            output = self.actors[mid](obs[aid], rnncs=self.rnncs[aid])  # [B, A]
            self.rnncs_[aid] = self.actors[mid].get_rnncs()
            if self.is_continuouss[aid]:
                mu, log_std = output  # [B, A]
                pi = td.Normal(mu, log_std.exp()).sample().tanh()
                mu.tanh_()  # squash mu  # [B, A]
            else:
                logits = output  # [B, A]
                mu = logits.argmax(-1)    # [B,]
                cate_dist = td.Categorical(logits=logits)
                pi = cate_dist.sample()  # [B,]
            action = pi if self._is_train_mode else mu
            acts_info[aid] = Data(action=action)
            actions[aid] = action
        return actions, acts_info

    @iton
    def _train(self, BATCH_DICT):
        '''
        TODO: Annotation
        '''
        summaries = defaultdict(dict)
        target_actions = {}
        target_log_pis = 1.
        for aid, mid in zip(self.agent_ids, self.model_ids):
            if self.is_continuouss[aid]:
                target_mu, target_log_std = self.actors[mid](BATCH_DICT[aid].obs_,
                                                             begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, A]
                dist = td.Independent(td.Normal(target_mu, target_log_std.exp()), 1)
                target_pi = dist.sample()   # [T, B, A]
                target_pi, target_log_pi = squash_action(
                    target_pi, dist.log_prob(target_pi).unsqueeze(-1))   # [T, B, A], [T, B, 1]
            else:
                target_logits = self.actors[mid](BATCH_DICT[aid].obs_,
                                                 begin_mask=BATCH_DICT['global'].begin_mask)    # [T, B, A]
                target_cate_dist = td.Categorical(logits=target_logits)
                target_pi = target_cate_dist.sample()   # [T, B]
                target_log_pi = target_cate_dist.log_prob(target_pi).unsqueeze(-1)  # [T, B, 1]
                target_pi = F.one_hot(target_pi, self.a_dims[aid]).float()  # [T, B, A]
            target_actions[aid] = target_pi
            target_log_pis *= target_log_pi

        target_log_pis += t.finfo().eps
        target_actions = t.cat(list(target_actions.values()), -1)   # [T, B, N*A]

        qs1, qs2, q_targets1, q_targets2 = {}, {}, {}, {}
        for mid in self.model_ids:
            qs1[mid] = self.critics[mid](
                [BATCH_DICT[id].obs for id in self.agent_ids],
                t.cat([BATCH_DICT[id].action for id in self.agent_ids], -1)
            )   # [T, B, 1]
            qs2[mid] = self.critics2[mid](
                [BATCH_DICT[id].obs for id in self.agent_ids],
                t.cat([BATCH_DICT[id].action for id in self.agent_ids], -1)
            )   # [T, B, 1]
            q_targets1[mid] = self.critics[mid].t([BATCH_DICT[id].obs_ for id in self.agent_ids],
                                                  target_actions)  # [T, B, 1]
            q_targets2[mid] = self.critics2[mid].t([BATCH_DICT[id].obs_ for id in self.agent_ids],
                                                   target_actions)  # [T, B, 1]

        q_loss = {}
        td_errors = 0.
        for aid, mid in zip(self.agent_ids, self.model_ids):
            q_target = t.minimum(q_targets1[mid], q_targets2[mid])  # [T, B, 1]
            dc_r = n_step_return(BATCH_DICT[aid].reward,
                                 self.gamma,
                                 BATCH_DICT[aid].done,
                                 q_target - self.alpha * target_log_pis,
                                 BATCH_DICT['global'].begin_mask).detach()  # [T, B, 1]
            td_error1 = qs1[mid] - dc_r   # [T, B, 1]
            td_error2 = qs2[mid] - dc_r   # [T, B, 1]
            td_errors += (td_error1 + td_error2) / 2
            q1_loss = td_error1.square().mean()    # 1
            q2_loss = td_error2.square().mean()    # 1
            q_loss[aid] = 0.5 * q1_loss + 0.5 * q2_loss
        summaries[aid].update(dict([
            ['Statistics/q_min', qs1[mid].min()],
            ['Statistics/q_mean', qs1[mid].mean()],
            ['Statistics/q_max', qs1[mid].max()]
        ]))
        self.critic_oplr.optimize(sum(q_loss.values()))

        log_pi_actions = {}
        log_pis = {}
        sample_pis = {}
        for aid, mid in zip(self.agent_ids, self.model_ids):
            if self.is_continuouss[aid]:
                mu, log_std = self.actors[mid](BATCH_DICT[aid].obs,
                                               begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, A]
                dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
                pi = dist.rsample()  # [T, B, A]
                pi, log_pi = squash_action(pi, dist.log_prob(pi).unsqueeze(-1))   # [T, B, A], [T, B, 1]
                pi_action = BATCH_DICT[aid].action.arctanh()
                _, log_pi_action = squash_action(pi_action, dist.log_prob(pi_action).unsqueeze(-1))    # [T, B, A], [T, B, 1]
            else:
                logits = self.actors[mid](BATCH_DICT[aid].obs,
                                          begin_mask=BATCH_DICT['global'].begin_mask)  # [T, B, A]
                logp_all = logits.log_softmax(-1)   # [T, B, A]
                gumbel_noise = td.Gumbel(0, 1).sample(logp_all.shape)   # [T, B, A]
                _pi = ((logp_all + gumbel_noise) / self.discrete_tau).softmax(-1)   # [T, B, A]
                _pi_true_one_hot = F.one_hot(_pi.argmax(-1), self.a_dims[aid]).float()  # [T, B, A]
                _pi_diff = (_pi_true_one_hot - _pi).detach()    # [T, B, A]
                pi = _pi_diff + _pi  # [T, B, A]
                log_pi = (logp_all * pi).sum(-1, keepdim=True)   # [T, B, 1]
                log_pi_action = (logp_all * BATCH_DICT[aid].action).sum(-1, keepdim=True)   # [T, B, 1]
            log_pi_actions[aid] = log_pi_action
            log_pis[aid] = log_pi
            sample_pis[aid] = pi

        actor_loss = {}
        for aid, mid in zip(self.agent_ids, self.model_ids):
            all_actions = {id: BATCH_DICT[id].action for id in self.agent_ids}
            all_actions[aid] = sample_pis[aid]
            all_log_pis = {id: log_pi_actions[id] for id in self.agent_ids}
            all_log_pis[aid] = log_pis[aid]

            q_s_pi = t.minimum(self.critics[mid]([BATCH_DICT[id].obs for id in self.agent_ids],
                                                 t.cat(list(all_actions.values()), -1),
                                                 begin_mask=BATCH_DICT['global'].begin_mask),
                               self.critics2[mid]([BATCH_DICT[id].obs for id in self.agent_ids],
                                                  t.cat(list(all_actions.values()), -1),
                                                  begin_mask=BATCH_DICT['global'].begin_mask))  # [T, B, 1]

            _log_pis = 1.
            for _log_pi in all_log_pis.values():
                _log_pis *= _log_pi
            _log_pis += t.finfo().eps
            actor_loss[aid] = -(q_s_pi - self.alpha * _log_pis).mean()  # 1

        self.actor_oplr.optimize(sum(actor_loss.values()))

        for aid in self.agent_ids:
            summaries[aid].update(dict([
                ['LOSS/actor_loss', actor_loss[aid]],
                ['LOSS/critic_loss', q_loss[aid]]
            ]))
        summaries['model'].update(dict([
            ['LOSS/actor_loss', sum(actor_loss.values())],
            ['LOSS/critic_loss', sum(q_loss.values())]
        ]))

        if self.auto_adaption:
            _log_pis = 1.
            _log_pis = 1.
            for _log_pi in log_pis.values():
                _log_pis *= _log_pi
            _log_pis += t.finfo().eps

            alpha_loss = - \
                (self.alpha * (_log_pis + self.target_entropy).detach()).mean()  # 1

            self.alpha_oplr.optimize(alpha_loss)
            summaries['model'].update([
                ['LOSS/alpha_loss', alpha_loss],
                ['LEARNING_RATE/alpha_lr', self.alpha_oplr.lr]
            ])
        return td_errors / self.n_agents_percopy, summaries

    def _after_train(self):
        super()._after_train()
        if self.annealing and not self.auto_adaption:
            self.log_alpha.copy_(self.alpha_annealing(self._cur_train_step).log())
        for critic in self.critics.values():
            critic.sync()
        for critic2 in self.critics2.values():
            critic2.sync()
