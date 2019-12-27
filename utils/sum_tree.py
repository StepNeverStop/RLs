import numpy as np


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
        assert capacity > 0, 'capacity must larger than zero'
        self.now = 0
        self.parent_node_count = self.get_parent_node_count(capacity)
        # print(self.parent_node_count)
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
        if self.now >= self.data[0]:
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

    def get_batch(self, ps):
        assert isinstance(ps, (list, np.ndarray))
        tidx, didx, p, d = zip(*[self.get(i) for i in ps])
        tidx, didx, p, d = map(np.asarray, [tidx, didx, p, d])
        d = [np.asarray(e) for e in zip(*d)]    # [[s, a], [s, a]] => [[s, s], [a, a]]
        return (tidx, didx, p, d)

    def _retrieve(self, tree_index, seg_p_total):
        left = 2 * tree_index
        right = left + 1
        # if index 0 is the root node
        # left = 2 * tree_index + 1
        # right = 2 * (tree_index + 1)
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
