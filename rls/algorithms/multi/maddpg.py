#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from torch import distributions as td
from typing import (List,
                    Union,
                    NoReturn,
                    Dict)

from rls.nn.represent_nets import (DefaultRepresentationNetwork,
                                   MultiAgentCentralCriticRepresentationNetwork)
from rls.algorithms.base.ma_off_policy import MultiAgentOffPolicy
from rls.utils.torch_utils import sync_params
from rls.nn.noised_actions import Noise_action_REGISTER
from rls.nn.models import (CriticQvalueOne,
                           ActorDct,
                           ActorDPG)
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


class MADDPG(MultiAgentOffPolicy):
    '''
    Multi-Agent Deep Deterministic Policy Gradient, https://arxiv.org/abs/1706.02275
    '''

    def __init__(self,
                 envspecs,

                 ployak=0.995,
                 noise_action='ou',
                 noise_params={
                     'sigma': 0.2
                 },
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 discrete_tau=1.0,
                 share_params=True,
                 network_settings={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'q': [32, 32]
                 },
                 **kwargs):
        '''
        TODO: Annotation
        '''
        super().__init__(envspecs=envspecs, **kwargs)
        self.ployak = ployak
        self.discrete_tau = discrete_tau
        self.share_params = share_params
        self.n_models_percopy = 1 if self.share_params else self.n_agents_percopy

        self.rep_nets = []
        self.target_rep_nets = []

        self.actors = []
        self.actor_targets = []
        self.critics = []
        self.critic_targets = []

        self.actor_oplrs = []
        self.critic_oplrs = []

        for i in range(self.n_models_percopy):
            rep_net = DefaultRepresentationNetwork(obs_spec=self.envspecs[i].obs_spec,
                                                   representation_net_params=self.representation_net_params).to(self.device)
            if self.envspecs[i].is_continuous:
                actor = ActorDPG(rep_net.h_dim,
                                 output_shape=self.envspecs[i].a_dim,
                                 network_settings=network_settings['actor_continuous']).to(self.device)
            else:
                actor = ActorDct(rep_net.h_dim,
                                 output_shape=self.envspecs[i].a_dim,
                                 network_settings=network_settings['actor_discrete']).to(self.device)
            critic = CriticQvalueOne(rep_net.h_dim*self.n_models_percopy,
                                     action_dim=sum([envspec.a_dim for envspec in self.envspecs]),
                                     network_settings=network_settings['q']).to(self.device)
            target_rep_net = deepcopy(rep_net)
            target_rep_net.eval()
            critic_target = deepcopy(critic)
            critic_target.eval()
            actor_target = deepcopy(actor)
            actor_target.eval()
            self.rep_nets.append(rep_net)
            self.critics.append(critic)
            self.target_rep_nets.append(target_rep_net)
            self.critic_targets.append(critic_target)
            self.actors.append(actor)
            self.actor_targets.append(actor_target)

            actor_oplr = OPLR(actor, actor_lr)
            critic_oplr = OPLR([rep_net, critic], critic_lr)
            self.actor_oplrs.append(actor_oplr)
            self.critic_oplrs.append(critic_oplr)

        for i in range(self.n_models_percopy):
            sync_params(self.actor_targets[i], self.actors[i])
            sync_params(self.critic_targets[i], self.critics[i])
            sync_params(self.target_rep_nets[i], self.rep_nets[i])

        # TODO: 添加动作类型判断
        self.noised_actions = [Noise_action_REGISTER[noise_action](**noise_params) for i in range(self.n_models_percopy)]

        self._worker_modules.update({f"repnet_{i}": self.rep_nets[i] for i in range(self.n_models_percopy)})
        self._worker_modules.update({f"actor_{i}": self.actors[i] for i in range(self.n_models_percopy)})
        self._worker_modules.update({f"critic_{i}": self.critics[i] for i in range(self.n_models_percopy)})

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update({f'actor_oplr-{i}': self.actor_oplrs[i] for i in range(self.n_models_percopy)})
        self._trainer_modules.update({f'critic_oplr-{i}': self.critic_oplrs[i] for i in range(self.n_models_percopy)})

        self.initialize_data_buffer()

    def reset(self):
        super().reset()
        for noised_action in self.noised_actions:
            noised_action.reset()

    def __call__(self, obs: List, evaluation=False):
        mus, pis = self.call(obs)
        return mus if evaluation else pis

    @iTensor_oNumpy
    def call(self, obs):
        mus = []
        pis = []
        for i in range(self.n_agents_percopy):
            j = 0 if self.share_params else i
            feat, _ = self.rep_nets[j](obs[i])
            output = self.actors[j](feat)
            if self.envspecs[i].is_continuous:
                mu = output
                pi = self.noised_actions[j](mu)
            else:
                logits = output
                mu = logits.argmax(1)
                cate_dist = td.Categorical(logits=logits)
                pi = cate_dist.sample()
            mus.append(mu)
            pis.append(pi)
        return mus, pis

    def _target_params_update(self):
        for i in range(self.n_models_percopy):
            sync_params(self.actor_targets[i], self.actors[i], self.ployak)
            sync_params(self.critic_targets[i], self.critics[i], self.ployak)
            sync_params(self.target_rep_nets[i], self.rep_nets[i], self.ployak)

    def learn(self, **kwargs) -> NoReturn:
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn()

    @iTensor_oNumpy
    def _train(self, BATCHs):
        '''
        TODO: Annotation
        '''
        summaries = {}
        target_actions = []
        feats = []
        feats_ = []
        for i in range(self.n_agents_percopy):
            j = 0 if self.share_params else i
            feat, _ = self.rep_nets[j](BATCHs[i].obs)
            feat_, _ = self.target_rep_nets[j](BATCHs[i].obs_)
            feats.append(feat)
            feats_.append(feat_)

            if self.envspecs[i].is_continuous:
                target_actions.append(self.actor_targets[j](feat_))
            else:
                target_logits = self.actor_targets[j](feat_)
                target_cate_dist = td.Categorical(logits=target_logits)
                target_pi = target_cate_dist.sample()
                action_target = t.nn.functional.one_hot(target_pi, self.envspecs[i].a_dim).float()
                target_actions.append(action_target)
        target_actions = t.cat(target_actions, -1)

        q_targets = []
        for i in range(self.n_agents_percopy):
            j = 0 if self.share_params else i
            q_targets.append(self.critic_targets[j](t.cat(feats_, -1), target_actions))

        q_loss = []
        for i in range(self.n_agents_percopy):
            j = 0 if self.share_params else i
            q = self.critics[j](
                t.cat(feats, -1),
                t.cat([BATCH.action for BATCH in BATCHs], -1)
            )
            dc_r = (BATCHs[i].reward + self.gamma * q_targets[i] * (1 - BATCHs[i].done)).detach()

            td_error = dc_r - q
            q_loss.append(0.5 * td_error.square().mean())
        if self.share_params:
            self.critic_oplrs[0].step(sum(q_loss))

        actor_loss = []
        feats = [feat.detach() for feat in feats]
        for i in range(self.n_agents_percopy):
            j = 0 if self.share_params else i

            if self.envspecs[i].is_continuous:
                mu = self.actors[j](feats[i])
            else:
                logits = self.actors[j](feats[i])
                logp_all = logits.log_softmax(-1)
                gumbel_noise = td.Gumbel(0, 1).sample(BATCHs[i].action.shape)
                _pi = ((logp_all + gumbel_noise) / self.discrete_tau).softmax(-1)
                _pi_true_one_hot = t.nn.functional.one_hot(_pi.argmax(-1), self.envspecs[i].a_dim).float()
                _pi_diff = (_pi_true_one_hot - _pi).detach()
                mu = _pi_diff + _pi

            q_actor = self.critics[j](
                t.cat(feats, -1),
                t.cat([BATCH.action for BATCH in BATCHs[:i]]+[mu]+[BATCH.action for BATCH in BATCHs[i+1:]], -1)
            )
            actor_loss.append(-q_actor.mean())
        if self.share_params:
            self.actor_oplrs[0].step(sum(actor_loss))

        # summaries[i] = dict([
        #     ['LOSS/actor_loss', actor_loss],
        #     ['LOSS/critic_loss', q_loss],
        #     ['Statistics/q_min', q.min()],
        #     ['Statistics/q_mean', q.mean()],
        #     ['Statistics/q_max', q.max()]
        # ])
        self.global_step.add_(1)
        return summaries
