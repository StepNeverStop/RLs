#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class Sum_Tree(object):
    def __init__(self, capacity):
        """
        capacity = 5，设置经验池大小
        tree = [0,1,2,3,4,5,6,7,8,9,10,11,12] 8-12存放叶子结点p值，1-7存放父节点、根节点p值的和，0存放树节点的数量
        data = [0,1,2,3,4] 0-4存放数据
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
        assert capacity > 0, 'capacity must larger than zero'
        self.capacity = capacity
        self.parent_node_count = self.get_parent_node_count(capacity)
        self.tree_data_offset = self.parent_node_count + 1
        # print(self.parent_node_count, self.tree_data_offset)
        self.reset()

    def reset(self):
        self.now = 0
        self._size = 0
        self.tree = np.zeros(self.tree_data_offset + self.capacity)
        self.tree[0] = len(self.tree) - 1   # 树的总节点数，也是最后一个节点的索引值
        self.data = np.zeros(self.capacity, dtype=object)

    def add(self, p, data):
        """
        p : property
        data : [s, a, r, s_, done]
        """
        self.data[self.now] = data
        tree_index = self.now + self.tree_data_offset
        self._updatetree(tree_index, p)
        self.now = (self.now + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(self, p, data):
        """
        p : property
        data : [[s, a, r, s_, done], ...]
        """
        num = len(data)
        idx = (np.arange(num) + self.now) % self.capacity   # [0, capacity-1]
        self.data[idx] = data
        tree_index = idx + self.tree_data_offset
        self._updatetree_batch(tree_index, p)
        self.now = (idx[-1] + 1) % self.capacity
        self._size = min(self._size + num, self.capacity)

    def _updatetree(self, tree_index, p):
        diff = p - self.tree[tree_index]
        self._propagate(tree_index, diff)
        self.tree[tree_index] = p

    def _updatetree_batch(self, tree_index, p):
        tree_index, idx = np.unique(tree_index, return_index=True)
        p = p[idx]
        diff = p - self.tree[tree_index]
        sort_index = np.argsort(tree_index)
        tree_index = np.sort(tree_index)
        diff = diff[sort_index]
        self._propagate_batch(tree_index, diff)
        self.tree[tree_index] = p

    def _propagate(self, tree_index, diff):
        parent = tree_index // 2
        self.tree[parent] += diff
        if parent != 1:
            self._propagate(parent, diff)

    def _propagate_batch(self, tree_index, diff):
        parent = tree_index // 2
        _parent, idx1, count = np.unique(parent, return_index=True, return_counts=True)
        _, idx2 = np.unique(parent[::-1], return_index=True)
        diff = (diff[- 1 - idx2] + diff[idx1]) * count / 2
        self.tree[_parent] += diff
        if (_parent != 1).all():
            self._propagate_batch(_parent, diff)

    def get(self, seg_p_total):
        """
        seg_p_total : The value of priority to sample
        """
        tree_index = self._retrieve(1, seg_p_total)
        data_index = tree_index - self.tree_data_offset
        return (tree_index, data_index, self.tree[tree_index], self.data[data_index])

    def get_batch(self, ps):
        assert isinstance(ps, (list, np.ndarray))
        tidx, didx, p, d = zip(*[self.get(i) for i in ps])
        tidx, didx, p, d = map(np.asarray, [tidx, didx, p, d])
        d = [np.asarray(e) for e in zip(*d)]    # [[s, a], [s, a]] => [[s, s], [a, a]]
        return (tidx, didx, p, d)

    def get_batch_parallel(self, ps):
        assert isinstance(ps, (list, np.ndarray))
        init_idx = np.full(len(ps), 1)
        tidx = self._retrieve_batch(init_idx, ps)
        didx = tidx - self.tree_data_offset
        p = self.tree[tidx]
        d = self.data[didx]
        tidx, didx, p, d = map(np.asarray, [tidx, didx, p, d])
        d = [np.asarray(e) for e in zip(*d)]    # [[s, a], [s, a]] => [[s, s], [a, a]]
        return (tidx, didx, p, d)

    def get_all(self):
        assert self._size > 0, 'no data in buffer now.'
        didx = np.arange(self._size)
        tidx = didx + self.tree_data_offset
        p = self.tree[tidx]
        d = self.data[didx]
        tidx, didx, p, d = map(np.asarray, [tidx, didx, p, d])
        d = [np.asarray(e) for e in zip(*d)]    # [[s, a], [s, a]] => [[s, s], [a, a]]
        return (tidx, didx, p, d)

    def get_all_exps(self):
        d = self.data[:self._size]
        d = [np.asarray(e) for e in zip(*d)]
        return d

    def _retrieve(self, tree_index, seg_p_total):
        left = 2 * tree_index
        right = left + 1
        # if index 0 is the root node
        # left = 2 * tree_index + 1
        # right = 2 * (tree_index + 1)
        if left >= self.tree[0]:
            return tree_index
        return self._retrieve(left, seg_p_total) if seg_p_total <= self.tree[left] else self._retrieve(right, seg_p_total - self.tree[left])

    def _retrieve_batch(self, tree_index, seg_p_total):
        left = 2 * tree_index
        right = left + 1
        if (left >= self.tree[0]).all():
            return tree_index
        # index = np.where(self.tree[left] >= seg_p_total, left, 0) + np.where(self.tree[left] < seg_p_total, right, 0)
        # seg_p_total = np.where(self.tree[left] >= seg_p_total, seg_p_total, 0) + np.where(self.tree[left] < seg_p_total, seg_p_total - self.tree[left], 0)
        index = np.where(seg_p_total < self.tree[left], left, right)
        seg_p_total = np.where(seg_p_total < self.tree[left], seg_p_total, seg_p_total - self.tree[left])
        return self._retrieve_batch(index, seg_p_total)

    def pp(self):
        print(self.tree, self.data)

    @property
    def total(self):
        return self.tree[1]

    def get_parent_node_count(self, capacity):
        i = 0
        while True:
            if pow(2, i) < capacity <= pow(2, i + 1):
                return pow(2, i + 1) - 1
            i += 1


if __name__ == "__main__":
    from time import time
    t = 10
    init_times = []
    sample_times = []
    update_times = []

    for i in range(t):
        tree = Sum_Tree(524288)
        # [[s, a], [s, a], ...524288]
        a = [[[1, 2], [3]] for _ in range(524288)]
        b = np.arange(524288) + 1
        start = time()
        tree.add_batch(b, a)
        init_times.append(time() - start)

        all_intervals = np.linspace(0, tree.total, 1024 + 1)
        ps = np.random.uniform(all_intervals[:-1], all_intervals[1:])
        start = time()
        tree.get_batch_parallel(ps)
        sample_times.append(time() - start)

        start = time()
        tree._updatetree_batch(np.random.randint(0, 524288, 1024), np.random.randint(0, 20, 1024))
        update_times.append(time() - start)

    # 0.24790284633636475 0.0028038501739501955 0.0010003089904785157
    print(np.asarray(init_times).mean(), np.asarray(sample_times).mean(), np.asarray(update_times).mean())

    init_times = []
    sample_times = []
    update_times = []

    for i in range(t):
        tree = Sum_Tree(524288)
        # [[s, a], [s, a], ...524288]
        a = [[[1, 2], [3]] for _ in range(524288)]
        b = np.arange(524288) + 1
        start = time()
        for _a, _b in zip(a, b):
            tree.add(_b, _a)
        init_times.append(time() - start)

        all_intervals = np.linspace(0, tree.total, 1024 + 1)
        ps = np.random.uniform(all_intervals[:-1], all_intervals[1:])
        start = time()
        tree.get_batch(ps)
        sample_times.append(time() - start)

        start = time()
        for _i, _p in zip(np.random.randint(0, 524288, 1024), np.random.randint(0, 20, 1024)):
            tree._updatetree(_i, _p)
        update_times.append(time() - start)

    # 5.809855628013611 0.01910092830657959 0.017893266677856446
    print(np.asarray(init_times).mean(), np.asarray(sample_times).mean(), np.asarray(update_times).mean())
