
import numpy as np

from typing import (List,
                    NoReturn)

from rls.memories.base_replay_buffer import ReplayBuffer
from rls.utils.specs import (BatchExperiences,
                             NamedTupleStaticClass)


class MultiAgentExperienceReplay(ReplayBuffer):
    def __init__(self,
                 n_agents: int,
                 batch_size: int,
                 capacity: int):
        super().__init__(batch_size, capacity)
        self._n_agents = n_agents
        self._data_pointer = 0
        self._buffers = np.empty((self._n_agents, capacity), dtype=object)

    def add(self, expss: List[BatchExperiences]) -> NoReturn:
        '''
        change [s, s],[a, a],[r, r] to [s, a, r],[s, a, r] and store every item in it.
        '''
        for exps in zip(*map(lambda x: list(NamedTupleStaticClass.unpack(x)), expss)):
            for i, exp in enumerate(exps):
                self._store_op(i, exp)
            self.update_rb_after_add()

        # for i, exps in enumerate(expss):
        #     for exp in NamedTupleStaticClass.unpack(exps):
        #         self._store_op(i, exp)

    def _store_op(self, i, exp: BatchExperiences) -> NoReturn:
        self._buffers[i, self._data_pointer] = exp
        # self.update_rb_after_add()

    def sample(self) -> BatchExperiences:
        '''
        change [[s, a, r],[s, a, r]] to [[s, s],[a, a],[r, r]]
        '''
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        idx = np.random.randint(0, self._size, n_sample)
        t = self._buffers[:, idx]
        return [NamedTupleStaticClass.pack(_t.tolist()) for _t in t]

    def get_all(self) -> BatchExperiences:
        return [NamedTupleStaticClass.pack(data.tolist()) for data in self._buffers[:, :self._size]]

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
        print(self._buffers[:, :])

    def save2hdf5(self):
        pass

    def loadhdf5(self):
        pass
