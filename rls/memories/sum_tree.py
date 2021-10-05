#!/usr/bin/env python3
# encoding: utf-8

import numpy as np


class Sum_Tree:

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
        self.parent_node_count = self._get_parent_node_count(capacity)
        self.tree_data_offset = self.parent_node_count + 1
        # print(self.parent_node_count, self.tree_data_offset)

        self._now = 0
        self._size = 0
        self.tree = np.zeros(self.tree_data_offset + self.capacity)
        self.tree[0] = len(self.tree) - 1  # 树的总节点数，也是最后一个节点的索引值

    def add_batch(self, p, n_step_delay=0):
        """
        p : property
        """
        p = p.ravel()
        B = p.shape[0]
        idx = (np.arange(B) + self._now) % self.capacity  # [0, capacity-1]
        # [parent_node_count+1, parent_node_count+capacity]
        tidx = idx + self.tree_data_offset
        if n_step_delay <= 0:
            self._updatetree_batch(tidx, p)
        else:
            self._updatetree_batch(tidx, np.zeros_like(p))
            # [0, capacity-1]
            _pre_idx = (idx - B * n_step_delay + self.capacity) % self.capacity
            p = np.where(_pre_idx < self._size, p, 0.)
            _pre_tidx = _pre_idx + self.tree_data_offset
            self._updatetree_batch(_pre_tidx, p)
        self._now = (self._now + B) % self.capacity
        self._size = min(self._size + B, self.capacity)

    def update_batch(self, didx, p):
        p = np.squeeze(p)
        tidx = didx + self.tree_data_offset
        self._updatetree_batch(tidx, p)

    def _updatetree_batch(self, tidx, p):
        tidx, idx = np.unique(tidx, return_index=True)
        p = p[idx]
        diff = p - self.tree[tidx]
        sort_index = np.argsort(tidx)
        tidx = np.sort(tidx)
        diff = diff[sort_index]
        self._propagate_batch(tidx, diff)
        self.tree[tidx] = p

    def _propagate_batch(self, tidx, diff):
        parent = tidx // 2
        _parent, idx1, count = np.unique(parent, return_index=True, return_counts=True)
        _, idx2 = np.unique(parent[::-1], return_index=True)
        diff = (diff[- 1 - idx2] + diff[idx1]) * count / 2
        self.tree[_parent] += diff
        if (_parent != 1).all():
            self._propagate_batch(_parent, diff)

    def get_batch_parallel(self, ps):
        assert isinstance(ps, (list, np.ndarray))
        init_idx = np.full(len(ps), 1)
        tidx = self._retrieve_batch(init_idx, ps)
        didx = tidx - self.tree_data_offset
        p = self.tree[tidx]
        return didx, p

    def _retrieve_batch(self, tidx, seg_p_total):
        left = 2 * tidx
        right = left + 1
        if (left >= self.tree[0]).all():
            return tidx
        # index = np.where(self.tree[left] >= seg_p_total, left, 0) + np.where(self.tree[left] < seg_p_total, right, 0)
        # seg_p_total = np.where(self.tree[left] >= seg_p_total, seg_p_total, 0) + np.where(self.tree[left] < seg_p_total, seg_p_total - self.tree[left], 0)
        index = np.where(seg_p_total < self.tree[left], left, right)
        seg_p_total = np.where(seg_p_total < self.tree[left], seg_p_total, seg_p_total - self.tree[left])
        return self._retrieve_batch(index, seg_p_total)

    @property
    def total(self):
        return self.tree[1]

    def _get_parent_node_count(self, capacity):
        i = 0
        while True:
            if pow(2, i) < capacity <= pow(2, i + 1):
                return pow(2, i + 1) - 1
            i += 1
