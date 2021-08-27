import numpy as np

from typing import Dict
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy

from rls.utils.np_utils import arrprint


class Recoder(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def episode_reset(self, episode):
        pass

    @abstractmethod
    def episode_step(self, rewards, dones):
        pass

    @abstractmethod
    def episode_end(self):
        pass


class SimpleMovingAverageRecoder(Recoder):

    def __init__(self,
                 n_copys,
                 agent_ids,
                 gamma=0.99,
                 verbose=False,
                 length=10):
        super().__init__()
        self.n_copys = n_copys
        self.agent_ids = agent_ids
        self.gamma = gamma
        self.verbose = verbose
        self.length = length

        self.now = 0
        self.r_list = []
        self.max = defaultdict(int)
        self.min = defaultdict(int)
        self.mean = defaultdict(int)

        self.total_step = 0
        self.episode = 0

    def episode_reset(self):
        self.steps = defaultdict(lambda: np.zeros((self.n_copys, ), dtype=int))
        self.total_returns = defaultdict(
            lambda: np.zeros((self.n_copys, ), dtype=float))
        self.discounted_returns = defaultdict(
            lambda: np.zeros((self.n_copys, ), dtype=float))
        self.already_dones = defaultdict(
            lambda: np.zeros((self.n_copys, ), dtype=bool))

    def episode_step(self, rewards: Dict[str, np.ndarray], dones: Dict[str, np.ndarray]):
        for id in self.agent_ids:
            self.total_step += 1
            self.discounted_returns[id] += (self.gamma ** self.steps[id]) * (
                1 - self.already_dones[id]) * rewards[id]
            self.steps[id] += (1 - self.already_dones[id]).astype(int)
            self.total_returns[id] += (1 -
                                       self.already_dones[id]) * rewards[id]
            self.already_dones[id] = np.logical_or(
                self.already_dones[id], dones[id])

    def episode_end(self):
        # TODO: optimize
        self.episode += 1
        self.r_list.append(deepcopy(self.total_returns))
        if self.now >= self.length:
            r_old = self.r_list.pop(0)
            for id in self.agent_ids:
                self.max[id] += (self.total_returns[id].max() -
                                 r_old[id].max()) / self.length
                self.min[id] += (self.total_returns[id].min() -
                                 r_old[id].min()) / self.length
                self.mean[id] += (self.total_returns[id].mean() -
                                  r_old[id].mean()) / self.length
        else:
            self.now = min(self.now + 1, self.length)
            for id in self.agent_ids:
                self.max[id] += (self.total_returns[id].max() -
                                 self.max[id]) / self.now
                self.min[id] += (self.total_returns[id].min() -
                                 self.min[id]) / self.now
                self.mean[id] += (self.total_returns[id].mean() -
                                  self.mean[id]) / self.now

    @property
    def is_all_done(self):  # TODO:
        if len(self.agent_ids) > 1:
            return np.logical_or(*self.already_dones.values()).all()
        else:
            return self.already_dones[self.agent_ids[0]].all()

    @property
    def has_done(self):  # TODO:
        if len(self.agent_ids) > 1:
            return np.logical_or(*self.already_dones.values()).any()
        else:
            return self.already_dones[self.agent_ids[0]].any()

    def summary_dict(self, title='Agent'):
        _dicts = {}
        for id in self.agent_ids:
            _dicts[id] = dict([
                [f'{title}/total_rt_mean', self.total_returns[id].mean()],
                [f'{title}/total_rt_min', self.total_returns[id].min()],
                [f'{title}/total_rt_max', self.total_returns[id].max()],
                [f'{title}/discounted_rt_mean',
                    self.discounted_returns[id].mean()],
                [f'{title}/discounted_rt_min', self.discounted_returns[id].min()],
                [f'{title}/discounted_rt_max', self.discounted_returns[id].max()],
                [f'{title}/sma_max', self.max[id]],
                [f'{title}/sma_min', self.min[id]],
                [f'{title}/sma_mean', self.mean[id]]
            ])
            if self.verbose:
                _dicts[id].update(dict([
                    [f'{title}/first_done_step', self.steps[id]
                        [self.already_dones[id] > 0].min() if self.has_done else -1],
                    [f'{title}/last_done_step', self.steps[id]
                        [self.already_dones[id] > 0].max() if self.has_done else -1]
                ]))
        return _dicts

    def __str__(self):
        _str = f'Eps: {self.episode:3d}'
        for id in self.agent_ids:
            _str += f'\n    Agent: {id.ljust(10)} | S: {self.steps[id].max():4d} | R: {arrprint(self.total_returns[id], 2)}'
            if self.verbose:
                first_done_step = self.steps[id][self.already_dones[id] > 0].min(
                ) if self.has_done else -1
                last_done_step = self.steps[id][self.already_dones[id] > 0].max(
                ) if self.has_done else -1
                _str += f' | FDS {first_done_step:4d} | LDS {last_done_step:4d}'
        return _str
