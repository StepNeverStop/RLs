import numpy as np

from abc import ABC, abstractmethod

from rls.utils.np_utils import arrprint


class Recoder(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def episode_reset(self, episode):
        pass

    @abstractmethod
    def step_update(self, rewards, dones):
        pass

    @abstractmethod
    def episode_end(self):
        pass


class SimpleMovingAverageRecoder(Recoder):

    def __init__(self,
                 n_copys,
                 n_agents,
                 gamma=0.99,
                 verbose=False,
                 length=10):
        super().__init__()
        self.n_copys = n_copys
        self.n_agents = n_agents
        self.gamma = gamma
        self.verbose = verbose
        self.total_step = 0
        self.length = length
        self.now = 0
        self.r_list = []
        self.max, self.min, self.mean = np.zeros(self.n_agents), np.zeros(self.n_agents), np.zeros(self.n_agents)
        self.episode_reset()

    def episode_reset(self, episode=0):
        self.episode = episode
        self.steps = np.zeros((self.n_agents, self.n_copys), dtype=int)
        self.total_returns = np.zeros((self.n_agents, self.n_copys), dtype=float)
        self.discounted_returns = np.zeros((self.n_agents, self.n_copys), dtype=float)
        self.already_dones = np.zeros((self.n_agents, self.n_copys))

    def step_update(self, rewards, dones):
        for i, (reward, done) in enumerate(zip(rewards, dones)):
            self.discounted_returns[i] += (self.gamma ** self.steps[i]) * (1 - self.already_dones[i]) * np.asarray(reward)
            self.total_step += 1
            self.steps[i] += (1 - self.already_dones[i]).astype(int)
            self.total_returns[i] += (1 - self.already_dones[i]) * np.asarray(reward)
            self.already_dones[i] = np.sign(self.already_dones[i] + np.asarray(done))

    def episode_end(self):
        # TODO: optimize
        self.r_list.append(self.total_returns.copy())
        if self.now >= self.length:
            r_old = self.r_list.pop(0)
            self.max += (self.total_returns.max(-1) - r_old.max(-1)) / self.length
            self.min += (self.total_returns.min(-1) - r_old.min(-1)) / self.length
            self.mean += (self.total_returns.mean(-1) - r_old.mean(-1)) / self.length
        else:
            self.now = min(self.now + 1, self.length)
            self.max += (self.total_returns.max(-1) - self.max) / self.now
            self.min += (self.total_returns.min(-1) - self.min) / self.now
            self.mean += (self.total_returns.mean(-1) - self.mean) / self.now

    @property
    def is_all_done(self):  # TODO:
        return all(self.already_dones.ravel())

    @property
    def has_done(self):
        return self.already_dones.any(-1)

    def summary_dict(self, title='Agent'):
        _dicts = {}
        for i in range(self.n_agents):
            _dicts[i] = dict([
                [f'{title}/total_rt_mean', self.total_returns[i].mean()],
                [f'{title}/total_rt_min', self.total_returns[i].min()],
                [f'{title}/total_rt_max', self.total_returns[i].max()],
                [f'{title}/discounted_rt_mean', self.discounted_returns[i].mean()],
                [f'{title}/discounted_rt_min', self.discounted_returns[i].min()],
                [f'{title}/discounted_rt_max', self.discounted_returns[i].max()],
                [f'{title}/sma_max', self.max[i]],
                [f'{title}/sma_min', self.min[i]],
                [f'{title}/sma_mean', self.mean[i]]
            ])
            if self.verbose:
                _dicts[i].update(dict([
                    [f'{title}/first_done_step', self.steps[i][self.already_dones[i] > 0].min() if self.has_done[i] else -1],
                    [f'{title}/last_done_step', self.steps[i][self.already_dones[i] > 0].max() if self.has_done[i] else -1]
                ]))
        return _dicts

    def __str__(self):
        _str = f'Eps: {self.episode:3d}'
        for i in range(self.n_agents):
            _str += f'\n    Agent: {i:3d} | S: {self.steps[i].max():4d} | R: {arrprint(self.total_returns[i], 2)}'
            if self.verbose:
                first_done_step = self.steps[i][self.already_dones[i] > 0].min() if self.has_done[i] else -1
                last_done_step = self.steps[i][self.already_dones[i] > 0].max() if self.has_done[i] else -1
                _str += f' | FDS {first_done_step:4d} | LDS {last_done_step:4d}'
        return _str
