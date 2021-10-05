from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Dict

import numpy as np

from rls.utils.np_utils import arrprint
from rls.utils.summary_collector import SummaryCollector


class Recoder(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def episode_reset(self):
        pass

    @abstractmethod
    def episode_step(self, rewards, dones):
        pass

    @abstractmethod
    def episode_end(self):
        pass


class SimpleMovingAverageRecoder(Recoder):

    def __init__(self,
                 n_copies,
                 agent_ids,
                 gamma=0.99,
                 verbose=False,
                 length=10):
        super().__init__()
        self._n_copies = n_copies
        self._agent_ids = agent_ids
        self._gamma = gamma
        self._verbose = verbose
        self._length = length

        self._now = 0
        self._r_list = []
        self._max = defaultdict(int)
        self._min = defaultdict(int)
        self._mean = defaultdict(int)

        self._total_step = 0
        self._episode = 0

        self._steps = None
        self._total_returns = None
        self._discounted_returns = None
        self._already_dones = None

        self._summary_collectors = {id: SummaryCollector(mode=SummaryCollector.ALL) for id in self._agent_ids}

    def episode_reset(self):
        self._steps = defaultdict(lambda: np.zeros((self._n_copies,), dtype=int))
        self._total_returns = defaultdict(lambda: np.zeros((self._n_copies,), dtype=float))
        self._discounted_returns = defaultdict(lambda: np.zeros((self._n_copies,), dtype=float))
        self._already_dones = defaultdict(lambda: np.zeros((self._n_copies,), dtype=bool))

    def episode_step(self, rewards: Dict[str, np.ndarray], dones: Dict[str, np.ndarray]):
        for id in self._agent_ids:
            self._total_step += 1
            self._discounted_returns[id] += (self._gamma ** self._steps[id]) * (1 - self._already_dones[id]) * rewards[
                id]
            self._steps[id] += (1 - self._already_dones[id]).astype(int)
            self._total_returns[id] += (1 - self._already_dones[id]) * rewards[id]
            self._already_dones[id] = np.logical_or(self._already_dones[id], dones[id])

    def episode_end(self):
        # TODO: optimize
        self._episode += 1
        self._r_list.append(deepcopy(self._total_returns))
        if self._now >= self._length:
            r_old = self._r_list.pop(0)
            for id in self._agent_ids:
                self._max[id] += (self._total_returns[id].max() - r_old[id].max()) / self._length
                self._min[id] += (self._total_returns[id].min() - r_old[id].min()) / self._length
                self._mean[id] += (self._total_returns[id].mean() - r_old[id].mean()) / self._length
        else:
            self._now = min(self._now + 1, self._length)
            for id in self._agent_ids:
                self._max[id] += (self._total_returns[id].max() - self._max[id]) / self._now
                self._min[id] += (self._total_returns[id].min() - self._min[id]) / self._now
                self._mean[id] += (self._total_returns[id].mean() - self._mean[id]) / self._now

    @property
    def is_all_done(self):  # TODO:
        if len(self._agent_ids) > 1:
            return np.logical_or(*self._already_dones.values()).all()
        else:
            return self._already_dones[self._agent_ids[0]].all()

    @property
    def has_done(self):  # TODO:
        if len(self._agent_ids) > 1:
            return np.logical_or(*self._already_dones.values()).any()
        else:
            return self._already_dones[self._agent_ids[0]].any()

    def summary_dict(self, scope='Agent'):
        for id in self._agent_ids:
            self._summary_collectors[id].add(scope, 'total_rt', self._total_returns[id])
            self._summary_collectors[id].add(scope, 'discounted_rt', self._discounted_returns[id])
            self._summary_collectors[id].add(scope, 'sma_rt', [self._min[id], self._mean[id], self._max[id]])
            if self._verbose:
                self._summary_collectors[id].add(scope, 'first_done_step', self._steps[id][
                    self._already_dones[id] > 0].min() if self.has_done else -1)
                self._summary_collectors[id].add(scope, 'last_done_step', self._steps[id][
                    self._already_dones[id] > 0].max() if self.has_done else -1)
        return {k: v.fetch() for k, v in self._summary_collectors.items()}

    def __str__(self):
        _str = f'Eps: {self._episode:3d}'
        for id in self._agent_ids:
            _str += f'\n    Agent: {id.ljust(10)} | S: {self._steps[id].max():4d} | R: {arrprint(self._total_returns[id], 2)}'
            if self._verbose:
                first_done_step = self._steps[id][self._already_dones[id] > 0].min() if self.has_done else -1
                last_done_step = self._steps[id][self._already_dones[id] > 0].max() if self.has_done else -1
                _str += f' | FDS {first_done_step:4d} | LDS {last_done_step:4d}'
        return _str
