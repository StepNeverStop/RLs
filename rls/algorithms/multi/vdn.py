#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy

from rls.algorithms.base.ma_off_policy import MultiAgentOffPolicy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import sync_params_list
from rls.nn.represent_nets import DefaultRepresentationNetwork
from rls.nn.models import CriticDueling
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy
from rls.nn.mixers import VDNMixer


class VDN(MultiAgentOffPolicy):
    '''
    Value-Decomposition Networks For Cooperative Multi-Agent Learning, http://arxiv.org/abs/1706.05296
    TODO: RNN, multi-step
    '''

    def __init__(self,
                 envspecs,

                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=2,
                 share_params=True,
                 network_settings={
                     'share': [128],
                     'v': [128],
                     'adv': [128]
                 },
                 **kwargs):
        assert not any([envspec.is_continuous for envspec in envspecs]), 'VDN only support discrete action space'
        super().__init__(envspecs=envspecs, **kwargs)
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval
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
            q_net = CriticDueling(rep_net.h_dim,
                                  output_shape=envspec.a_dim,
                                  network_settings=network_settings).to(self.device)
            target_rep_net = deepcopy(rep_net)
            target_rep_net.eval()
            q_target_net = deepcopy(q_net)
            q_target_net.eval()
            return rep_net, q_net, target_rep_net, q_target_net

        rets = [build_nets(self.envspecs[i]) for i in range(self.n_models_percopy)]
        self.rep_nets, self.q_nets, self.target_rep_nets, self.q_target_nets = tuple(zip(*rets))
        self.mixer = VDNMixer()
        self.target_mixer = VDNMixer()

        sync_params_list([self.target_rep_nets+self.q_target_nets+(self.target_mixer,),
                          self.rep_nets+self.q_nets+(self.mixer,)])

        self.oplr = OPLR(self.q_nets+self.rep_nets+(self.mixer,), lr)

        self._worker_modules.update({f"repnet_{i}": self.rep_nets[i] for i in range(self.n_models_percopy)})
        self._worker_modules.update({f"model_{i}": self.q_nets[i] for i in range(self.n_models_percopy)})

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(mixer=self.mixer)
        self._trainer_modules.update(oplr=self.oplr)
        self.initialize_data_buffer()

    def __call__(self, obs, evaluation=False):
        actions = self.call(obs, evaluation)
        return actions

    @iTensor_oNumpy  # TODO: optimization
    def call(self, obs, evaluation):
        actions = []
        for i, j in zip(self.agents_indexs, self.models_indexs):
            if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
                actions.append(np.random.randint(0, self.envspecs[i].a_dim, self.n_copys))
            else:
                feat, _ = self.rep_nets[j](obs[i])
                q_values = self.q_nets[j](feat)
                actions.append(q_values.argmax(-1))
        return actions

    def _target_params_update(self):
        if self.global_step % self.assign_interval == 0:
            sync_params_list([self.target_rep_nets+self.q_target_nets+(self.target_mixer,),
                              self.rep_nets+self.q_nets+(self.mixer,)])

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn()

    @iTensor_oNumpy
    def _train(self, BATCHs):
        summaries = {}
        reward = t.zeros_like(BATCHs[0].reward)
        done = t.zeros_like(BATCHs[0].done)
        q_evals = []
        q_target_next_maxs = []
        for i, j in zip(self.agents_indexs, self.models_indexs):
            reward += BATCHs[i].reward
            done += BATCHs[i].done
            feat, _ = self.rep_nets[j](BATCHs[i].obs)
            feat_, _ = self.target_rep_nets[j](BATCHs[i].obs_)
            feat__, _ = self.rep_nets[j](BATCHs[i].obs_)

            q = self.q_nets[j](feat)
            next_q = self.q_nets[j](feat__)
            q_target = self.q_target_nets[j](feat_)

            q_eval = (q * BATCHs[i].action).sum(1, keepdim=True)
            q_evals.append(q_eval)
            next_max_action = next_q.argmax(1)
            next_max_action_one_hot = t.nn.functional.one_hot(next_max_action.squeeze(), self.envspecs[i].a_dim).float()

            q_target_next_max = (q_target * next_max_action_one_hot).sum(1, keepdim=True)
            q_target_next_maxs.append(q_target_next_max)
        q_eval_all = self.mixer(q_evals)
        q_target_next_max_all = self.target_mixer(q_target_next_maxs)
        q_target_all = (reward + self.gamma * q_target_next_max_all * (1 - done > 0)).detach()
        td_error = q_target_all - q_eval_all
        q_loss = td_error.square().mean()
        self.oplr.step(q_loss)

        self.global_step.add_(1)
        summaries['model'] = dict([
            ['LOSS/loss', q_loss],
            ['Statistics/q_max', q_eval_all.max()],
            ['Statistics/q_min', q_eval_all.min()],
            ['Statistics/q_mean', q_eval_all.mean()]
        ])
        return summaries
