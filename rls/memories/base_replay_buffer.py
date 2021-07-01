
from abc import ABC, abstractmethod
from typing import (Any,
                    NoReturn,
                    Union,
                    List,
                    Tuple,
                    Optional)

from rls.utils.specs import BatchExperiences


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
