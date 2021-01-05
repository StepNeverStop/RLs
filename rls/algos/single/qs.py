#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.plot import (ion,
                            ioff,
                            plot_heatmap)
from rls.utils.specs import BatchExperiences


class QS:
    '''
    Q-learning/Sarsa/Expected Sarsa.
    '''

    def __init__(self,
                 envspec,

                 mode='q',
                 lr=0.2,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 **kwargs):
        assert not envspec.is_continuous
        self.mode = mode
        self.s_dim = envspec.s_dim
        self.a_dim = envspec.a_dim
        self.n_agents = envspec.n_agents
        self.gamma = float(kwargs.get('gamma', 0.999))
        self.max_train_step = int(kwargs.get('max_train_step', 1000))
        self.step = 0
        self.train_step = 0
        if self.n_agents <= 0:
            raise ValueError('agents num must larger than zero.')
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.table = np.zeros(shape=(self.s_dim, self.a_dim))
        self.lr = lr
        self.next_a = np.zeros(self.n_agents, dtype=np.int32)
        self.mask = []
        ion()

    def one_hot2int(self, x):
        idx = [np.where(np.asarray(i))[0][0] for i in x]
        return idx

    def partial_reset(self, done):
        self.mask = np.where(done)[0]

    def choose_action(self, obs, evaluation=False):
        s = self.one_hot2int(obs.flatten_vector())
        if self.mode == 'q':
            return self._get_action(s, evaluation)
        elif self.mode == 'sarsa' or self.mode == 'expected_sarsa':
            a = self._get_action(s, evaluation)
            self.next_a[self.mask] = a[self.mask]
            return self.next_a

    def _get_action(self, s, evaluation=False, _max=False):
        a = np.array([np.argmax(self.table[i, :]) for i in s])
        if not _max:
            if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
                a = np.random.randint(0, self.a_dim, self.n_agents)
        return a

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

    def store_data(self, exps: BatchExperiences):
        self.step += 1
        s = self.one_hot2int(exps.obs.flatten_vector())
        s_ = self.one_hot2int(exps.obs_.flatten_vector())
        if self.mode == 'q':
            a_ = self._get_action(s_, _max=True)
            value = self.table[s_, a_]
        else:
            self.next_a = self._get_action(s_)
            if self.mode == 'expected_sarsa':
                value = np.mean(self.table[s_, :], axis=-1)
            else:
                value = self.table[s_, self.next_a]
        self.table[s, exps.action] = (1 - self.lr) * self.table[s, exps.action] + self.lr * (exps.reward + self.gamma * (1 - exps.done) * value)
        if self.step % 1000 == 0:
            plot_heatmap(self.s_dim, self.a_dim, self.table)

    def close(self):
        ioff()

    def no_op_store(self, exps: BatchExperiences):
        pass

    def __getattr__(self, x):
        # print(x)
        return lambda *args, **kwargs: 0
