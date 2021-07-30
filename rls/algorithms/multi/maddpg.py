#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from collections import defaultdict
from torch import distributions as td
from typing import (List,
                    Union,
                    NoReturn,
                    Dict)

from rls.nn.represent_nets import (DefaultRepresentationNetwork,
                                   MultiAgentCentralCriticRepresentationNetwork)
from rls.algorithms.base.ma_off_policy import MultiAgentOffPolicy
from rls.utils.torch_utils import sync_params_list
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
        self.share_params = share_params and self._is_envspecs_all_equal
        self.agents_indexs = list(range(self.n_agents_percopy))
        if self.share_params:
            self.n_models_percopy = 1
            self.models_indexs = [0] * self.n_agents_percopy
        else:
            self.n_models_percopy = self.n_agents_percopy
            self.models_indexs = self.agents_indexs

        def build_nets(envspec):
            rep_net = DefaultRepresentationNetwork(obs_spec=envspec.obs_spec,
                                                   representation_net_params=self.representation_net_params).to(self.device)
            target_rep_net = deepcopy(rep_net)
            target_rep_net.eval()

            if envspec.is_continuous:
                actor = ActorDPG(rep_net.h_dim,
                                 output_shape=envspec.a_dim,
                                 network_settings=network_settings['actor_continuous']).to(self.device)
            else:
                actor = ActorDct(rep_net.h_dim,
                                 output_shape=envspec.a_dim,
                                 network_settings=network_settings['actor_discrete']).to(self.device)
            actor_target = deepcopy(actor)
            actor_target.eval()

            critic = CriticQvalueOne(rep_net.h_dim*self.n_agents_percopy,
                                     action_dim=sum([envspec.a_dim for envspec in self.envspecs]),
                                     network_settings=network_settings['q']).to(self.device)
            critic_target = deepcopy(critic)
            critic_target.eval()
            return (rep_net, actor, critic, target_rep_net, actor_target, critic_target)

        rets = [build_nets(self.envspecs[i]) for i in range(self.n_models_percopy)]
        self.rep_nets, self.actors, self.critics, self.target_rep_nets, \
            self.actor_targets, self.critic_targets = tuple(zip(*rets))

        self.actor_oplr = OPLR(self.actors, actor_lr)
        self.critic_oplr = OPLR(self.rep_nets+self.critics, critic_lr)

        sync_params_list([self.rep_nets+self.actors+self.critics,
                         self.target_rep_nets+self.actor_targets+self.critic_targets])

        # TODO: 添加动作类型判断
        self.noised_actions = [Noise_action_REGISTER[noise_action](**noise_params) for i in range(self.n_models_percopy)]

        self._worker_modules.update({f"repnet_{i}": self.rep_nets[i] for i in range(self.n_models_percopy)})
        self._worker_modules.update({f"actor_{i}": self.actors[i] for i in range(self.n_models_percopy)})
        self._worker_modules.update({f"critic_{i}": self.critics[i] for i in range(self.n_models_percopy)})

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(actor_oplr=self.actor_oplr)
        self._trainer_modules.update(critic_oplr=self.critic_oplr)

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
        for i, j in zip(self.agents_indexs, self.models_indexs):
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
        sync_params_list([self.rep_nets+self.actors+self.critics,
                         self.target_rep_nets+self.actor_targets+self.critic_targets])

    def learn(self, **kwargs) -> NoReturn:
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn()

    @iTensor_oNumpy
    def _train(self, BATCHs):
        '''
        TODO: Annotation
        '''
        summaries = defaultdict(dict)
        target_actions = []
        feats = []
        feats_ = []
        for i, j in zip(self.agents_indexs, self.models_indexs):
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

        q_loss = []
        for i, j in zip(self.agents_indexs, self.models_indexs):
            q_target = self.critic_targets[j](t.cat(feats_, -1), target_actions)
            q = self.critics[j](
                t.cat(feats, -1),
                t.cat([BATCH.action for BATCH in BATCHs], -1)
            )
            dc_r = (BATCHs[i].reward + self.gamma * q_target * (1 - BATCHs[i].done)).detach()
            td_error = dc_r - q
            q_loss.append(0.5 * td_error.square().mean())
            summaries[i].update(dict([
                ['Statistics/q_min', q.min()],
                ['Statistics/q_mean', q.mean()],
                ['Statistics/q_max', q.max()]
            ]))
        self.critic_oplr.step(sum(q_loss))

        actor_loss = []
        feats = [feat.detach() for feat in feats]
        for i, j in zip(self.agents_indexs, self.models_indexs):
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

        self.actor_oplr.step(sum(actor_loss))

        for i in self.agents_indexs:
            summaries[i].update(dict([
                ['LOSS/actor_loss', actor_loss[i]],
                ['LOSS/critic_loss', q_loss[i]],
                # ['Statistics/q_min', q.min()],
                # ['Statistics/q_mean', q.mean()],
                # ['Statistics/q_max', q.max()]
            ]))
        summaries['model'].update(dict([
            ['LOSS/actor_loss', sum(actor_loss)],
            ['LOSS/critic_loss', sum(q_loss)]
        ]))
        self.global_step.add_(1)
        return summaries
