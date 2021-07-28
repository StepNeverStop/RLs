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
                 gamma=0.99,
                 verbose=False,
                 length=10):
        super().__init__()
        self.n_copys = n_copys
        self.gamma = gamma
        self.verbose = verbose
        self.total_step = 0
        self.length = length
        self.now = 0
        self.r_list = []
        self.max, self.min, self.mean = 0, 0, 0
        self.episode_reset()

    def episode_reset(self, episode=0):
        self.episode = episode
        self.steps = np.zeros(self.n_copys, dtype=int)
        self.total_returns = np.zeros(self.n_copys, dtype=float)
        self.discounted_returns = np.zeros(self.n_copys, dtype=float)
        self.already_dones = np.zeros(self.n_copys)

    def step_update(self, rewards, dones):
        self.discounted_returns += (self.gamma ** self.steps) * (1 - self.already_dones) * np.asarray(rewards)
        self.total_step += 1
        self.steps += (1 - self.already_dones).astype(int)
        self.total_returns += (1 - self.already_dones) * np.asarray(rewards)
        self.already_dones = np.sign(self.already_dones + np.asarray(dones))

    def episode_end(self):
        # TODO: optimize
        self.r_list.append(self.total_returns.copy())
        if self.now >= self.length:
            r_old = self.r_list.pop(0)
            self.max += (self.total_returns.max() - r_old.max()) / self.length
            self.min += (self.total_returns.min() - r_old.min()) / self.length
            self.mean += (self.total_returns.mean() - r_old.mean()) / self.length
        else:
            self.now = min(self.now + 1, self.length)
            self.max += (self.total_returns.max() - self.max) / self.now
            self.min += (self.total_returns.min() - self.min) / self.now
            self.mean += (self.total_returns.mean() - self.mean) / self.now

    @property
    def is_all_done(self):
        return all(self.already_dones)

    @property
    def has_done(self):
        return any(self.already_dones)

    @property
    def summary_dict(self):
        _dict = dict(
            total_rt_mean=self.total_returns.mean(),
            total_rt_min=self.total_returns.min(),
            total_rt_max=self.total_returns.max(),
            discounted_rt_mean=self.discounted_returns.mean(),
            discounted_rt_min=self.discounted_returns.min(),
            discounted_rt_max=self.discounted_returns.max(),
            sma_max=self.max,
            sma_min=self.min,
            sma_mean=self.mean
        )
        if self.verbose:
            _dict.update(dict(
                first_done_step=self.steps[self.already_dones > 0].min() if self.has_done else -1,
                last_done_step=self.steps[self.already_dones > 0].max() if self.has_done else -1
            ))
        return _dict

    def __str__(self):
        _str = f'Eps: {self.episode:3d} | S: {self.steps.max():4d} | R: {arrprint(self.total_returns, 2)}'
        if self.verbose:
            first_done_step = self.steps[self.already_dones > 0].min() if self.has_done else -1
            last_done_step = self.steps[self.already_dones > 0].max() if self.has_done else -1
            _str += f' | FDS {first_done_step:4d} | LDS {last_done_step:4d}'
        return _str


class SimpleMovingAverageMultiAgentRecoder(Recoder):

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
        self.max, self.min, self.mean = 0, 0, 0
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
    def is_all_done(self):
        return all(self.already_dones.ravel())

    @property
    def has_done(self):
        return any(self.already_dones.ravel())

    @property
    def summary_dict(self):
        _dicts = {}
        for i in range(self.n_agents):
            _dicts[i] = dict([
                ['AGENT/total_rt_mean', self.total_returns[i].mean()],
                ['AGENT/total_rt_min', self.total_returns[i].min()],
                ['AGENT/total_rt_max', self.total_returns[i].max()],
                ['AGENT/discounted_rt_mean', self.discounted_returns[i].mean()],
                ['AGENT/discounted_rt_min', self.discounted_returns[i].min()],
                ['AGENT/discounted_rt_max', self.discounted_returns[i].max()],
                ['AGENT/sma_max', self.max[i]],
                ['AGENT/sma_min', self.min[i]],
                ['AGENT/sma_mean', self.mean[i]]
            ])
            # # TODO: check
            # if self.verbose:
            #     _dicts[i].update(dict([
            #         'AGENT/first_done_step':self.steps[self.already_dones[i] > 0].min() if self.has_done else -1,
            #         'AGENT/last_done_step':self.steps[self.already_dones[i] > 0].max() if self.has_done else -1
            #     ]))
        return _dicts

    def __str__(self):
        _str = f'Eps: {self.episode:3d}'
        for i in range(self.n_agents):
            _str += f'\nAgent: {i:3d} | S: {self.steps[i].max():4d} | R: {arrprint(self.total_returns[i], 2)}'
        # if self.verbose:
        #     first_done_step = self.steps[self.already_dones > 0].min() if self.has_done else -1
        #     last_done_step = self.steps[self.already_dones > 0].max() if self.has_done else -1
        #     _str += f' | FDS {first_done_step:4d} | LDS {last_done_step:4d}'
        return _str
