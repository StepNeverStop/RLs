#!/usr/bin/env python3
# encoding: utf-8

import sys
import numpy as np
import tensorflow as tf

from abc import ABC, abstractmethod
from typing import (Any,
                    NoReturn,
                    Union,
                    List,
                    Tuple,
                    Optional)

from rls.memories.sum_tree import Sum_Tree
from rls.utils.specs import (BatchExperiences,
                             NamedTupleStaticClass)


class ReplayBuffer(ABC):
    def __init__(self,
                 batch_size: int,
                 capacity: int):
        assert isinstance(batch_size, int) and batch_size >= 0, 'batch_size must be int and larger than 0'
        assert isinstance(capacity, int) and capacity >= 0, 'capacity must be int and larger than 0'
        self.batch_size = batch_size
        self.capacity = capacity
        self._size = 0

    def reset(self):
        self._size = 0

    @abstractmethod
    def sample(self) -> Any:
        pass

    @abstractmethod
    def add(self, exps: BatchExperiences) -> Any:
        pass

    def is_empty(self) -> bool:
        return self._size == 0

    def update(self, *args) -> Any:
        pass


class ExperienceReplay(ReplayBuffer):
    def __init__(self,
                 batch_size: int,
                 capacity: int):
        super().__init__(batch_size, capacity)
        self._data_pointer = 0
        self._buffer = np.empty(capacity, dtype=object)

    def add(self, exps: BatchExperiences) -> NoReturn:
        '''
        change [s, s],[a, a],[r, r] to [s, a, r],[s, a, r] and store every item in it.
        '''
        for exp in NamedTupleStaticClass.unpack(exps):
            self._store_op(exp)

    def _store_op(self, exp: BatchExperiences) -> NoReturn:
        self._buffer[self._data_pointer] = exp
        self.update_rb_after_add()

    def sample(self) -> BatchExperiences:
        '''
        change [[s, a, r],[s, a, r]] to [[s, s],[a, a],[r, r]]
        '''
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        t = np.random.choice(self._buffer[:self._size], size=n_sample, replace=False)
        # return [np.asarray(e) for e in zip(*t)]
        return NamedTupleStaticClass.pack(t.tolist())

    def get_all(self) -> BatchExperiences:
        return NamedTupleStaticClass.pack(self._buffer[:self._size].tolist())

    def update_rb_after_add(self) -> NoReturn:
        self._data_pointer += 1
        if self._data_pointer >= self.capacity:  # replace when exceed the capacity
            self._data_pointer = 0
        if self._size < self.capacity:
            self._size += 1

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_lg_batch_size(self) -> bool:
        return self._size > self.batch_size

    @property
    def show_rb(self) -> NoReturn:
        print('RB size: ', self._size)
        print('RB capacity: ', self.capacity)
        print(self._buffer[:])


class PrioritizedExperienceReplay(ReplayBuffer):
    '''
    This PER will introduce some bias, 'cause when the experience with the minimum probability has been collected, the min_p that be updated may become inaccuracy.
    '''

    def __init__(self,
                 batch_size: int,
                 capacity: int,
                 max_train_step: int,
                 alpha: float,
                 beta: float,
                 epsilon: float,
                 global_v: bool):
        '''
        inputs:
            max_train_step: use for calculating the decay interval of beta
            alpha: control sampling rule, alpha -> 0 means uniform sampling, alpha -> 1 means complete td_error sampling
            beta: control importance sampling ratio, beta -> 0 means no IS, beta -> 1 means complete IS.
            epsilon: a small positive number that prevents td-error of 0 from never being replayed.
            global_v: whether using the global
        '''
        assert epsilon > 0, 'epsilon must larger than zero'
        super().__init__(batch_size, capacity)
        self.tree = Sum_Tree(capacity)
        self.alpha = alpha
        self.beta = self.init_beta = beta
        self.beta_interval = (1. - beta) / max_train_step
        self.epsilon = epsilon
        self.IS_w = 1   # weights of variables by using Importance Sampling
        self.global_v = global_v
        self.reset()

    def reset(self):
        self.tree.reset()
        super().reset()
        self.beta = self.init_beta
        self.min_p = sys.maxsize
        self.max_p = np.power(self.epsilon, self.alpha)

    def add(self, exps: BatchExperiences) -> NoReturn:
        '''
        input: [ss, visual_ss, as, rs, s_s, visual_s_s, dones]
        '''
        self.add_batch(list(NamedTupleStaticClass.unpack(exps)))
        # for data in NamedTupleStaticClass.unpack(exps):
        #     self._store_op(data)

    def _store_op(self, data: BatchExperiences) -> NoReturn:
        self.tree.add(self.max_p, data)
        if self._size < self.capacity:
            self._size += 1

    def add_batch(self, data: List[BatchExperiences]) -> NoReturn:
        num = len(data)
        self.tree.add_batch(np.full(num, self.max_p), data)
        self._size = min(self._size + num, self.capacity)

    def apex_add_batch(self, td_error, *args):
        data = list(zip(*args))
        num = len(data)
        prios = np.power(np.abs(td_error) + self.epsilon, self.alpha)
        self.tree.add_batch(prios, data)
        self._size = min(self._size + num, self.capacity)

    def sample(self, return_index: bool = False) -> Union[List, Tuple]:
        '''
        output: weights, [ss, visual_ss, as, rs, s_s, visual_s_s, dones]
        '''
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        all_intervals = np.linspace(0, self.tree.total, n_sample + 1)
        ps = np.random.uniform(all_intervals[:-1], all_intervals[1:])
        idxs, data_indx, p, data = self.tree.get_batch_parallel(ps)
        self.last_indexs = idxs
        _min_p = self.min_p if self.global_v and self.min_p < sys.maxsize else p.min()
        self.IS_w = np.power(_min_p / p, self.beta)
        data = NamedTupleStaticClass.pack(data.tolist())
        if return_index:
            return data, idxs
        else:
            return data

    def get_all(self, return_index: bool = False) -> BatchExperiences:
        idxs, data_indx, p, data = self.tree.get_all()
        self.last_indexs = idxs
        _min_p = self.min_p if self.global_v and self.min_p < sys.maxsize else p.min()
        self.IS_w = np.power(_min_p / p, self.beta)
        data = NamedTupleStaticClass.pack(data.tolist())
        if return_index:
            return data, idxs
        else:
            return data

    def get_all_exps(self):
        return self.tree.get_all_exps()

    @property
    def is_lg_batch_size(self) -> bool:
        return self._size > self.batch_size

    def update(self,
               priority: Union[List, np.ndarray],
               index: Optional[Union[List, np.ndarray]] = None) -> NoReturn:
        '''
        input: priorities
        '''
        assert hasattr(priority, '__len__'), 'priority must have attribute of len()'
        idxs = index if index is not None else self.last_indexs
        assert len(priority) == len(idxs), 'length between priority and last_indexs must equal'
        self.beta = min(self.beta + self.beta_interval, 1.)
        priority = np.power(np.abs(priority) + self.epsilon, self.alpha)
        self.min_p = min(self.min_p, priority.min())
        self.max_p = max(self.max_p, priority.max())
        self.tree._updatetree_batch(idxs, priority)
        # [self.tree._updatetree(idx, p) for idx, p in zip(idxs, priority)]

    def get_IS_w(self) -> np.ndarray:
        return self.IS_w

    @property
    def size(self) -> int:
        return self._size


class NStepWrapper:
    def __init__(self,
                 buffer: ReplayBuffer,
                 gamma: float,
                 n: int,
                 agents_num: int):
        '''
        gamma: discount factor
        n: n step
        agents_num: batch experience
        '''
        self.buffer = buffer
        self.n = n
        self.gamma = gamma
        self.agents_num = agents_num
        self.queue = [[] for _ in range(agents_num)]

    def add(self, exps: BatchExperiences) -> NoReturn:
        for i, data in enumerate(NamedTupleStaticClass.unpack(exps)):
            self._per_store(i, data)

    def _per_store(self, i: int, data: BatchExperiences) -> NoReturn:
        # TODO: 优化
        q = self.queue[i]
        if len(q) == 0:  # 如果Nstep临时经验池为空，就直接添加
            q.append(data)
            return

        if len(q) == self.n:
            self._store_op(q.pop(0))
        if not NamedTupleStaticClass.check_equal(q[-1].obs_, data.obs):    # 如果截断了，非常规done，把Nstep临时经验池中已存在的经验都存进去，临时经验池清空
            q.clear()   # 保证经验池中不存在不足N长度的序列，有done的除外，因为（1-done）为0，导致gamma的次方计算不准确也没有关系。
            q.append(data)
        else:
            _len = len(q)
            for j in range(_len):   # 然后再存入一条最新的经验到Nstep临时经验池
                q[j] = q[j]._replace(reward=q[j].reward + data.reward * (self.gamma ** (_len - j)))
                q[j] = q[j]._replace(obs_=data.obs_)
                q[j] = q[j]._replace(done=data.done)
            q.append(data)
            if data.done:  # done or not # 如果新数据是done，就清空临时经验池
                while q:    # (1-done)会清零不正确的n-step
                    self._store_op(q.pop())

    def __getattr__(self, name):
        return getattr(self.buffer, name)


class NStepExperienceReplay(NStepWrapper):
    '''
    Replay Buffer + NStep
    '''

    def __init__(self,
                 batch_size: int,
                 capacity: int,
                 gamma: float,
                 n: int,
                 agents_num: int):
        super().__init__(
            buffer=ExperienceReplay(batch_size, capacity),
            gamma=gamma, n=n, agents_num=agents_num
        )


class NStepPrioritizedExperienceReplay(NStepWrapper):
    '''
    PER + NStep
    '''

    def __init__(self,
                 batch_size: int,
                 capacity: int,
                 max_train_step: int,
                 alpha: float,
                 beta: float,
                 epsilon: float,
                 global_v: bool,
                 gamma: float,
                 n: int,
                 agents_num: int):
        super().__init__(
            buffer=PrioritizedExperienceReplay(batch_size, capacity, max_train_step, alpha, beta, epsilon, global_v),
            gamma=gamma, n=n, agents_num=agents_num
        )


class EpisodeExperienceReplay(ReplayBuffer):

    def __init__(self,
                 batch_size: int,
                 capacity: int,
                 agents_num: int,
                 burn_in_time_step: int,
                 train_time_step: int):
        super().__init__(batch_size, capacity)
        self.agents_num = agents_num
        self.burn_in_time_step = burn_in_time_step
        self.train_time_step = train_time_step
        self.timestep = burn_in_time_step + train_time_step
        self.queue = [[] for _ in range(agents_num)]
        self._data_pointer = 0
        self._buffer = np.empty(capacity, dtype=object)

    def add(self, exps: BatchExperiences) -> NoReturn:
        '''
        change [s, s],[a, a],[r, r] to [s, a, r],[s, a, r] and store every item in it.
        '''
        for i, data in enumerate(NamedTupleStaticClass.unpack(exps)):
            self._per_store(i, data)

    def _per_store(self, i: int, data: BatchExperiences) -> NoReturn:
        q = self.queue[i]
        if len(q) == 0:
            q.append(data)
            return
        if not NamedTupleStaticClass.check_equal(q[-1].obs_, data.obs):
            self._store_op(q.copy())
            q.clear()
            q.append(data)
            return
        if data.done:
            q.append(data)
            self._store_op(q.copy())
            q.clear()
            return
        q.append(data)

    def _store_op(self, data: List[BatchExperiences]) -> NoReturn:
        self._buffer[self._data_pointer] = data
        self.update_rb_after_add()

    def update_rb_after_add(self) -> NoReturn:
        self._data_pointer += 1
        if self._data_pointer >= self.capacity:  # replace when exceed the capacity
            self._data_pointer = 0
        if self._size < self.capacity:
            self._size += 1

    def sample(self) -> BatchExperiences:
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        trajs = np.random.choice(self._buffer[:self._size], size=n_sample, replace=False)   # 选n_sample条轨迹

        def f(v, l):    # [B, T, N]
            return lambda x: tf.keras.preprocessing.sequence.pad_sequences(x, padding='pre', dtype='float32', value=v, maxlen=l, truncating='pre')

        def truncate(traj):
            idx = np.random.randint(max(1, len(traj) - self.timestep + 1))  # [min, max)
            return traj[idx:idx + self.timestep]

        datas = []  # [B, 不定长时间步, N]
        for traj in trajs:
            data = NamedTupleStaticClass.pack(truncate(traj))
            datas.append(data)

        sample_data = NamedTupleStaticClass.pack(datas)
        sample_data = NamedTupleStaticClass.data_convert(f(v=1., l=self.timestep), sample_data, ['done'])   # [B, T, N]
        sample_data = NamedTupleStaticClass.data_convert(f(v=0., l=self.timestep), sample_data)  # [B, T, N]

        burn_in_data = NamedTupleStaticClass.data_convert(lambda x: x[:, :self.burn_in_time_step], sample_data)
        train_data = NamedTupleStaticClass.data_convert(lambda x: x[:, self.burn_in_time_step:], sample_data)

        self.burn_in_data = NamedTupleStaticClass.data_convert(lambda x: tf.reshape(x, [-1, *x.shape[2:]]), burn_in_data)
        train_data = NamedTupleStaticClass.data_convert(lambda x: tf.reshape(x, [-1, *x.shape[2:]]), train_data)

        return train_data

    def get_burn_in_data(self) -> BatchExperiences:
        return self.burn_in_data

    @property
    def is_full(self) -> bool:
        return self._size == self.capacity

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_lg_batch_size(self) -> bool:
        return self._size > self.batch_size

    @property
    def show_rb(self) -> NoReturn:
        print('RB size: ', self._size)
        print('RB capacity: ', self.capacity)
        print(self._buffer)


if __name__ == "__main__":
    buff = EpisodeExperienceReplay(4, 10, 2)
