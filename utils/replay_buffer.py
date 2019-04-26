import math
import numpy as np


class ReplayBuffer(object):
    _data_pointer = 0
    _size = 0

    def __init__(self, batch_size, capacity):
        self.batch_size = batch_size
        self.capacity = capacity

        self._buffer = np.empty(capacity, dtype=object)

    def add(self, *args):
        for arg in args:
            assert len(arg.shape) == 2
            assert len(arg) == len(args[0])

        for i in range(len(args[0])):
            self._buffer[self._data_pointer] = tuple(arg[i] for arg in args)
            self._data_pointer += 1

            if self._data_pointer >= self.capacity:  # replace when exceed the capacity
                self._data_pointer = 0

            if self._size < self.capacity:
                self._size += 1

    def sample(self):
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        t = np.random.choice(self._buffer[:self._size], size=n_sample, replace=False)
        return [np.array(e) for e in zip(*t)]

    @property
    def is_full(self):
        return self._size == self.capacity

    @property
    def size(self):
        return self._size

    @property
    def is_lg_batch_size(self):
        return self._size > self.batch_size


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Story data with its priority in the tree.
    """
    _data_pointer = 0
    _size = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self._tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self._data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self._data_pointer + self.capacity - 1
        self._data[self._data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self._data_pointer += 1
        if self._data_pointer >= self.capacity:  # replace when exceed the capacity
            self._data_pointer = 0

        if self._size < self.capacity:
            self._size += 1

    def update(self, tree_idx, p):
        change = p - self._tree[tree_idx]
        self._tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self._tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self._tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self._tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self._tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self._tree[leaf_idx], self._data[data_idx]

    @property
    def total_p(self):
        return self._tree[0]  # the root

    @property
    def max(self):
        if self._size == 0:
            return 0
        return np.max(self._tree[self.capacity - 1:self._size + self.capacity - 1])

    @property
    def min(self):
        return np.min(self._tree[self.capacity - 1:self._size + self.capacity - 1])

    @property
    def size(self):
        return self._size


class PrioritizedReplayBuffer(object):  # stored as ( s, a, r, s_ ) in SumTree
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, batch_size, capacity):
        self.batch_size = batch_size
        capacity = 2**math.floor(math.log2(capacity))
        self._tree = SumTree(capacity)

    def add(self, *args):
        max_p = self._tree.max
        if max_p == 0:
            max_p = self.abs_err_upper

        for i in range(len(args[0])):
            self._tree.add(max_p, tuple(arg[i] for arg in args))

    def sample(self):
        n_sample = self.batch_size if self.is_lg_batch_size else self.size

        points, transitions, is_weights = np.empty((n_sample,), dtype=np.int32), np.empty((n_sample,), dtype=object), np.empty((n_sample, 1))
        pri_seg = self._tree.total_p / n_sample       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = self._tree.min / self._tree.total_p     # for later calculate ISweight

        for i in range(n_sample):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self._tree.get_leaf(v)
            prob = p / self._tree.total_p
            is_weights[i, 0] = np.power(prob / min_prob, -self.beta)
            points[i], transitions[i] = idx, data
        return points, tuple(np.array(e) for e in zip(*transitions)), is_weights

    def update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)

        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self._tree.update(ti, p)

    @property
    def size(self):
        return self._tree.size

    @property
    def is_lg_batch_size(self):
        return self.size > self.batch_size
