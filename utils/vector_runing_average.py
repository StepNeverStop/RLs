import numpy as np


class DefaultRunningAverage:

    def __init__(self, m=0.0, v=0.0, n=0):
        self._mean = m
        self._var = v
        self._n = n

    def __call__(self, x):
        pass

    def var(self):
        return 1.

    def mean(self):
        return 0.

    def std(self):
        return np.sqrt(self.var())

    def normalize(self, x):
        return (x - self.mean()) / self.std()


class SimpleRunningAverage(DefaultRunningAverage):

    def __init__(self, m=0.0, v=0.0, n=0, dim=1):
        m = np.zeros(shape=(dim,))
        v = np.zeros(shape=(dim,))
        super().__init__(m, v, n)

    def __call__(self, x):
        if isinstance(x, (np.ndarray, list, tuple)):
            for _x in x:
                self.update(_x)
        else:
            self.update(x)

    def update(self, x):
        self._n += 1
        new_mean = self._mean + (x - self._mean) / self._n
        new_var = self._var + (x - self._mean) * (x - new_mean)
        self._mean = new_mean
        self._var = new_var

    def var(self):
        assert self._n > 0
        return self._var / self._n

    def mean(self):
        return self._mean


if __name__ == "__main__":

    dra = DefaultRunningAverage()
    print(dra.mean(), dra.std(), dra.var())

    sra = SimpleRunningAverage(dim=4)

    for i in range(100):
        sra(np.full(4, i))
        if i % 10 == 0:
            print(sra.mean(), sra.std(), sra.var())
    print(sra.mean(), sra.std(), sra.var())

    print(sra.normalize(np.full(4, 49.5)))

    # a = np.random.randn(10, 5, 2)
    # b = np.random.randn(5, 2)
    # print((a/b).shape)

