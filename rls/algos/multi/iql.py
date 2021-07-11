#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy

from rls.algos.base.ma_off_policy import MultiAgentOffPolicy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.torch_utils import sync_params
from rls.nn.represent_nets import DefaultRepresentationNetwork
from rls.nn.models import CriticQvalueAll
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


class IQL(MultiAgentOffPolicy):

    def __init__(self,
                 envspecs,

                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=2,
                 share_params=True,
                 network_settings: List[int] = [32, 32],
                 **kwargs):
        assert not any([envspec.is_continuous for envspec in envspecs]), 'IQL only support discrete action space'
        super().__init__(envspecs=envspecs, **kwargs)
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval
        self.share_params = share_params
        self.n_models_percopy = 1 if self.share_params else self.n_agents_percopy

        self.rep_nets = []
        self.q_nets = []
        self.target_rep_nets = []
        self.q_target_nets = []
        self.oplrs = []
        for i in range(self.n_models_percopy):
            rep_net = DefaultRepresentationNetwork(obs_spec=self.envspecs[i].obs_spec,
                                                   representation_net_params=self.representation_net_params)
            q_net = CriticQvalueAll(rep_net.h_dim,
                                    output_shape=self.envspecs[i].a_dim,
                                    network_settings=network_settings)
            target_rep_net = deepcopy(rep_net)
            target_rep_net.eval()
            q_target_net = deepcopy(q_net)
            q_target_net.eval()
            self.rep_nets.append(ep_net)
            self.q_nets.append(q_net)
            self.target_rep_nets.append(target_rep_net)
            self.q_target_nets.append(q_target_net)
            oplr = OPLR([rep_net, q_net], lr)
            self.oplrs.append(oplr)

        for i in range(self.n_models_percopy):
            sync_params(self.target_rep_nets[i], self.rep_nets[i])
            sync_params(self.q_target_nets[i], self.q_nets[i])

        self._worker_modules.update({f"repnet_{i}": self.rep_nets[i] for i in range(self.n_models_percopy)})
        self._worker_modules.update({f"model_{i}": self.q_nets[i] for i in range(self.n_models_percopy)})

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update({f"oplr_{i}": oplr for oplr in self.oplrs})
        self.initialize_data_buffer()

    @iTensor_oNumpy
    def __call__(self, obs, evaluation=False):
        actions = []
        for i in range(self.n_agents_percopy):
            j = 0 if self.share_params else i
            if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
                actions.append(np.random.randint(0, self.envspecs[i].a_dim, self.n_copys))
            else:
                :
                feat, _ = self.rep_nets[j](obs[i])
                q_values = self.q_nets[j](feat)
                actions.append(q_values.argmax(-1))
        return actions

    def _target_params_update(self):
        if self.global_step % self.assign_interval == 0:
            for i in range(self.n_models_percopy):
                sync_params(self.target_rep_nets[i], self.rep_nets[i])
                sync_params(self.q_target_nets[i], self.q_nets[i])

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn()

    @iTensor_oNumpy
    def _train(self, BATCHs):
        summaries = {}
        for i in range(self.n_agents_percopy):
            j = 0 if self.share_params else i
            feat, _ = self.rep_nets[j](BATCHs[i].obs)
            feat_, _ = self.target_rep_nets[j](BATCHs[i].obs_)
            q = self.q_nets[j](feat)
            q_next = self.q_target_nets[j](feat_)
            q_eval = (q * BATCHs[i].action).sum(1, keepdim=True)
            q_target = (BATCHs[i].reward + self.gamma * (1 - BATCHs[i].done) * q_next.max(1, keepdim=True)[0]).detach()
            td_error = q_target - q_eval
            q_loss = td_error.square().mean()
            self.oplrs[j].step(q_loss)
            # TODO:
            summaries[i] = dict([
                ['LOSS/loss', q_loss],
                ['Statistics/q_max', q_eval.max()],
                ['Statistics/q_min', q_eval.min()],
                ['Statistics/q_mean', q_eval.mean()]
            ])
        self.global_step.add_(1)
        return summaries
