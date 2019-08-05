import numpy as np
from abc import ABC, abstractmethod


class Buffer(ABC):
    @abstractmethod
    def sample(self) -> list:
        pass


class ReplayBuffer(Buffer):
    def __init__(self, batch_size, capacity):
        self.batch_size = batch_size
        self.capacity = capacity
        self._data_pointer = 0
        self._size = 0
        self._buffer = np.empty(capacity, dtype=object)

    def add(self, *args):
        '''
        change [[s, s],[a, a],[r, r]] to [[s, a, r],[s, a, r]] and store every item in it.
        '''
        if hasattr(args[0], '__len__'):
            for i in range(len(args[0])):
                self._buffer[self._data_pointer] = tuple(arg[i] for arg in args)
                self.update_rb_after_add()
        else:
            self._buffer[self._data_pointer] = args
            self.update_rb_after_add()

    def sample(self):
        '''
        change [[s, a, r],[s, a, r]] to [[s, s],[a, a],[r, r]]
        '''
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        t = np.random.choice(self._buffer[:self._size], size=n_sample, replace=False)
        return [np.array(e) for e in zip(*t)]

    def update_rb_after_add(self):
        self._data_pointer += 1
        if self._data_pointer >= self.capacity:  # replace when exceed the capacity
            self._data_pointer = 0
        if self._size < self.capacity:
            self._size += 1

    @property
    def is_full(self):
        return self._size == self.capacity

    @property
    def size(self):
        return self._size

    @property
    def is_lg_batch_size(self):
        return self._size > self.batch_size

    @property
    def show_rb(self):
        print('RB size: ', self._size)
        print('RB capacity: ', self.capacity)
        print(self._buffer[:, np.newaxis])


class Sum_Tree(object):
    def __init__(self, capacity):
        """
        capacity = 5，设置经验池大小
        tree = [0,1,2,3,4,5,6,7,8,9,10,11,12] 8-12存放叶子结点p值，1-7存放父节点、根节点p值的和，0存放树节点的数量
        data = [0,1,2,3,4,5] 1-5存放数据， 0存放capacity
        Tree structure and array storage:
        Tree index:
                    1         -> storing priority sum
              /          \ 
             2            3
            / \          / \
          4     5       6   7
         / \   / \     / \  / \
        8   9 10   11 12                   -> storing priority for transitions
        """
        assert capacity > 0
        self.now = 0
        self.parent_node_count = self.get_parent_node_count(capacity)
        print(self.parent_node_count)
        self.tree = np.zeros(self.parent_node_count + capacity + 1)
        self.tree[0] = len(self.tree) - 1
        self.data = np.zeros(capacity + 1, dtype=object)
        self.data[0] = capacity

    def add(self, p, data):
        """
        p : property
        data : [s, a, r, s_, done]
        """
        tree_index = self.now + self.parent_node_count + 1
        self.data[self.now + 1] = data
        self._updatetree(tree_index, p)
        self.now += 1
        if self.now > self.data[0]:
            self.now = 0

    def _updatetree(self, tree_index, p):
        diff = p - self.tree[tree_index]
        self._propagate(tree_index, diff)
        self.tree[tree_index] = p

    def _propagate(self, tree_index, diff):
        parent = tree_index // 2
        self.tree[parent] += diff
        if parent != 1:
            self._propagate(parent, diff)

    @property
    def total(self):
        return self.tree[1]

    def get(self, seg_p_total):
        """
        seg_p_total : The value of priority to sample
        """
        tree_index = self._retrieve(1, seg_p_total)
        data_index = tree_index - self.parent_node_count
        return (tree_index, data_index, self.tree[tree_index], self.data[data_index])

    def _retrieve(self, tree_index, seg_p_total):
        left = 2 * tree_index
        right = left + 1
#         left = 2 * tree_index + 1
#         right = 2 * (tree_index + 1)
        if left >= self.tree[0]:
            return tree_index
        return self._retrieve(left, seg_p_total) if seg_p_total <= self.tree[left] else self._retrieve(right, seg_p_total - self.tree[left])

    def pp(self):
        print(self.tree, self.data)

    def get_parent_node_count(self, capacity):
        i = 0
        while True:
            if pow(2, i) < capacity <= pow(2, i + 1):
                return pow(2, i + 1) - 1
            i += 1


class PrioritizedReplayBuffer(Buffer):
    def __init__(self, batch_size, capacity, alpha, beta, epsilon):
        self.batch_size = batch_size
        self.capacity = capacity
        self._size = 0
        self.alpha = alpha
        self.beta = beta
        self.tree = Sum_Tree(capacity)
        self.epsilon = epsilon
        self.min_p = np.inf

    def add(self, p, *args):
        '''
        input: priorities, [ss, as, rs, _ss, dones]
        '''
        p = np.power(np.abs(p) + self.epsilon, self.alpha)
        min_p = p.min()
        if min_p < self.min_p:
            self.min_p = min_p
        if hasattr(args[0], '__len__'):
            for i in range(len(args[0])):
                self.tree.add(p[i], tuple(arg[i] for arg in args))
                if self._size < self.capacity:
                    self._size += 1
        else:
            self.tree.add(p, args)
            if self._size < self.capacity:
                self._size += 1

    def sample(self):
        '''
        output: weights, [ss, as, rs, _ss, dones]
        '''
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        interval = self.tree.total / n_sample
        segment = [self.tree.total - i * interval for i in range(n_sample + 1)]
        t = [self.tree.get(np.random.uniform(segment[i], segment[i + 1], 1)) for i in range(n_sample)]
        t = [np.array(e) for e in zip(*t)]
        self.last_indexs = t[0]
        return np.power(self.min_p / t[-2], self.beta), t[-1]

    @property
    def is_lg_batch_size(self):
        return self._size > self.batch_size

    def update_priority(self, priority):
        '''
        input: priorities
        '''
        assert hasattr(priority, '__len__')
        assert len(priority) == len(self.last_indexs)
        for i in range(len(priority)):
            self.tree._updatetree(self.last_indexs[i], priority[i])
