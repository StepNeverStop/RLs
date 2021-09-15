from abc import ABC, abstractmethod


class Base(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def random_action(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def episode_reset(self):
        pass

    @abstractmethod
    def episode_step(self):
        pass

    @abstractmethod
    def episode_end(self):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def resume(self):
        pass

    @property
    @abstractmethod
    def still_learn(self):
        pass
