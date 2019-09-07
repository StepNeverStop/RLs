import numpy as np
from .sum_tree import Sum_Tree
from abc import ABC, abstractmethod

# [s, visual_s, a, r, s_, visual_s_, done] must be this format.


class ReplayBuffer(ABC):
    def __init__(self, batch_size, capacity):
        assert type(batch_size) == int and batch_size > 0
        assert type(capacity) == int and capacity > 0
        self.batch_size = batch_size
        self.capacity = capacity
        self._size = 0

    @abstractmethod
    def sample(self) -> list:
        pass

    @abstractmethod
    def add(self, *args) -> None:
        pass

    def is_empty(self):
        return self._size == 0

    def update(self, *args) -> None:
        pass


class EpisodeExperienceReplay(ReplayBuffer):
    # TODO: implement padding so that makes each episode has the same length.
    def __init__(self, batch_size, capacity):
        super().__init__(batch_size, capacity)
        self._data_pointer = 0
        self.agents_num = 0
        self._buffer = np.empty(capacity, dtype=object)
        for i in range(self._buffer.shape[0]):
            self._buffer[i] = ExperienceReplay(1000, 1000)

    def done(self):
        self._data_pointer = (self._data_pointer + self.agents_num) % self.capacity
        for i in range(self.agents_num):
            self._buffer[(self._data_pointer + i) % self.capacity] = ExperienceReplay(1000, 1000)
        self._size += self.agents_num
        if self._size > self.capacity:
            self._size = self.capacity

    def add(self, *args):
        '''
        change [s, s],[a, a],[r, r] to [s, a, r],[s, a, r] and store every item in it.
        '''
        if hasattr(args[0], '__len__') or hasattr(args[1], '__len__'):
            self.agents_num = len(args[0])
            for i in range(len(args[0])):
                self._store_op(i, list(arg[i] for arg in args))
        else:
            self.agents_num = 1
            self._store_op(i, args)

    def _store_op(self, i, data):
        self._buffer[(self._data_pointer + i) % self.capacity]._store_op(data)

    def sample(self):
        '''
        return [[[s,s,...],[a,a,...],[r,r,...]],
                [[s,s,...],[a,a,...],[r,r,...]],
                ...]
        '''
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        t = np.random.choice(self._buffer[:self._size], size=n_sample, replace=False)
        return [i.get_all() for i in t]

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
        print('Episode RB size: ', self._size)
        print('Episode RB capacity: ', self.capacity)
        for i in self._buffer:
            i.show_rb


class ExperienceReplay(ReplayBuffer):
    def __init__(self, batch_size, capacity):
        super().__init__(batch_size, capacity)
        self._data_pointer = 0
        self._buffer = np.empty(capacity, dtype=object)

    def add(self, *args):
        '''
        change [s, s],[a, a],[r, r] to [s, a, r],[s, a, r] and store every item in it.
        '''
        if hasattr(args[0], '__len__') or hasattr(args[1], '__len__'):
            for i in range(len(args[0])):
                self._store_op(list(arg[i] for arg in args))
        else:
            self._store_op(args)

    def _store_op(self, data):
        self._buffer[self._data_pointer] = data
        self.update_rb_after_add()

    def sample(self):
        '''
        change [[s, a, r],[s, a, r]] to [[s, s],[a, a],[r, r]]
        '''
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        t = np.random.choice(self._buffer[:self._size], size=n_sample, replace=False)
        return [np.array(e) for e in zip(*t)]

    def get_all(self):
        return [np.array(e) for e in zip(*self._buffer[:self._size])]

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


class PrioritizedExperienceReplay(ReplayBuffer):
    '''
    This PER will introduce some bias, 'cause when the experience with the minimum probability has been collected, the min_p that be updated may become inaccuracy.
    '''

    def __init__(self, batch_size, capacity, max_episode, alpha, beta, epsilon):
        assert epsilon > 0
        super().__init__(batch_size, capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_interval = (1 - beta) / max_episode
        self.tree = Sum_Tree(capacity)
        self.epsilon = epsilon
        self.min_p = epsilon
        self.max_p = epsilon

    def add(self, *args):
        '''
        input: [ss, visual_ss, as, rs, s_s, visual_s_s, dones]
        '''
        if hasattr(args[0], '__len__') or hasattr(args[1], '__len__'):
            for i in range(len(args[0])):
                self._store_op(list(arg[i] for arg in args))
        else:
            self._store_op(args)

    def _store_op(self, data):
        self.tree.add(self.max_p, data)
        if self._size < self.capacity:
            self._size += 1

    def sample(self):
        '''
        output: weights, [ss, visual_ss, as, rs, s_s, visual_s_s, dones]
        '''
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        interval = self.tree.total / n_sample
        segment = [self.tree.total - i * interval for i in range(n_sample + 1)]
        t = [self.tree.get(np.random.uniform(segment[i], segment[i + 1], 1)) for i in range(n_sample)]
        t = [np.array(e) for e in zip(*t)]
        d = [np.array(e) for e in zip(*t[-1])]
        self.last_indexs = t[0]
        return np.power(self.min_p / t[-2], self.beta), d

    @property
    def is_lg_batch_size(self):
        return self._size > self.batch_size

    def update(self, priority, episode):
        '''
        input: priorities
        '''
        assert hasattr(priority, '__len__')
        assert len(priority) == len(self.last_indexs)
        self.beta += self.beta_interval * episode
        priority = np.power(np.abs(priority) + self.epsilon, self.alpha)
        min_p = priority.min()
        max_p = priority.max()
        if min_p < self.min_p:
            self.min_p = min_p
        if max_p > self.max_p:
            self.max_p = max_p
        for i in range(len(priority)):
            self.tree._updatetree(self.last_indexs[i], priority[i])


class NStepExperienceReplay(ExperienceReplay):
    '''
    [s, visual_s, a, r, s_, visual_s_, done] must be this format.
    '''

    def __init__(self, batch_size, capacity, gamma, n, agents_num):
        super().__init__(batch_size, capacity)
        self.n = n
        self.gamma = gamma
        self.exps_pointer = np.zeros(agents_num, dtype=np.int32)
        self.exps = [[()] * n for i in range(agents_num)]

    def add(self, *args):
        '''
        change [s, s],[a, a],[r, r] to [s, a, r],[s, a, r] and store every item in it.
        '''
        if hasattr(args[0], '__len__') or hasattr(args[1], '__len__'):
            for i in range(len(args[0])):
                self._store_op(list(arg[i] for arg in args), i)

        else:
            self._store_op(args)

    def _store_op(self, data, i=0):
        '''
        这段代码真是烂透了啊啊啊啊！！！！
        '''
        # if self.exps_pointer[i] > 0 and self.exps[i][self.exps_pointer[i] - 1][3] != data[0]:
        if self.exps_pointer[i] > 0 and ((data[0] != self.exps[i][self.exps_pointer[i] - 1][4]).any() or data[1] != self.exps[i][self.exps_pointer[i] - 1][5]):
            # if self.exps_pointer[i] > 0 and all([(val == data[0][i]).all() for i, val in enumerate(self.exps[i][self.exps_pointer[i] - 1][3])]):  # 因为data[0]代表状态s，由列表[np.array, np.array]组成，所以比较这样一个列表十分麻烦
            # 判断是因为done结束的episode，还是因为超过了max_step。如果是达到了max_step就执行下边的程序
            # 通过判断经验是不是第一个，而且判断上一条经验的下一个状态与该条经验的状态是否相同，如果不同，说明episode断了，就将临时经验池中的先存入
            for k in range(self.exps_pointer[i]):
                self.exps[i][k][-3:] = self.exps[i][self.exps_pointer[i] - 1][-3:]
                self._buffer[self._data_pointer] = self.exps[i][k]
                self.update_rb_after_add()
            self.exps[i] = [()] * self.n
            self.exps_pointer[i] = 0
        self.exps[i][self.exps_pointer[i]] = data  # 存入临时经验池
        for j in range(self.exps_pointer[i]):
            # 根据n_step和折扣因子gamma给之前经验的奖励进行加和
            self.exps[i][j][3] += pow(self.gamma, self.exps_pointer[i] - j) * data[3]
        if data[-1]:
            # 判断该经验的done_flag是True还是False，如果是True，就执行下边的程序
            # 把临时经验池中所有的经验都存入
            for k in range(self.exps_pointer[i] + 1):
                self.exps[i][k][-3:] = data[-3:]
                self._buffer[self._data_pointer] = self.exps[i][k]
                self.update_rb_after_add()
            self.exps[i] = [()] * self.n
            self.exps_pointer[i] = 0
        elif self.exps_pointer[i] == self.n - 1:
            # 如果没done，但是达到了临时经验池的长度，即n，则把最前边的经验存入， 并把之后的经验向前移动一位
            self.exps[i][0][-3:] = data[-3:]
            self._buffer[self._data_pointer] = self.exps[i][0]
            self.update_rb_after_add()
            del self.exps[i][0]
            self.exps[i].append(())
        else:
            # 如果没done，临时经验池也没满，就把指针后移
            self.exps_pointer[i] += 1


class NStepPrioritizedExperienceReplay(PrioritizedExperienceReplay):
    '''
    [s, visual_s, a, r, s_, visual_s_, done] must be this format.
    '''

    def __init__(self, batch_size, capacity, max_episode, gamma, alpha, beta, epsilon, agents_num, n):
        super().__init__(batch_size, capacity, alpha, beta, epsilon, max_episode)
        self.n = n
        self.gamma = gamma
        self.exps_pointer = np.zeros(agents_num, dtype=np.int32)
        self.exps = [[()] * n for i in range(agents_num)]

    def add(self, *args):
        '''
        input: [ss, visual_ss, as, rs, s_s, visual_s_s, dones]
        '''
        if hasattr(args[0], '__len__') or hasattr(args[1], '__len__'):
            for i in range(len(args[0])):
                self._store_op(list(arg[i] for arg in args), i)
        else:
            self._store_op(args)

    def _store_op(self, data, i=0):
        '''
        这段代码真是烂透了啊啊啊啊！！！！
        '''
        if self.exps_pointer[i] > 0 and ((data[0] != self.exps[i][self.exps_pointer[i] - 1][4]).any() or data[1] != self.exps[i][self.exps_pointer[i] - 1][5]):
            # 判断是因为done结束的episode，还是因为超过了max_step。如果是达到了max_step就执行下边的程序
            # 通过判断经验是不是第一个，而且判断上一条经验的下一个状态与该条经验的状态是否相同，如果不同，说明episode断了，就将临时经验池中的先存入
            for k in range(self.exps_pointer[i]):
                self.exps[i][k][-3:] = self.exps[i][self.exps_pointer[i] - 1][-3:]
                self.tree.add(self.max_p, self.exps[i][k])
                if self._size < self.capacity:
                    self._size += 1
            self.exps[i] = [()] * self.n
            self.exps_pointer[i] = 0
        self.exps[i][self.exps_pointer[i]] = data  # 存入临时经验池
        for j in range(self.exps_pointer[i]):
            # 根据n_step和折扣因子gamma给之前经验的奖励进行加和
            self.exps[i][j][3] += pow(self.gamma, self.exps_pointer[i] - j) * data[3]
        if data[-1]:
            # 判断该经验的done_flag是True还是False，如果是True，就执行下边的程序
            # 把临时经验池中所有的经验都存入
            for k in range(self.exps_pointer[i] + 1):
                self.exps[i][k][-3:] = data[-3:]
                self.tree.add(self.max_p, self.exps[i][k])
                if self._size < self.capacity:
                    self._size += 1
            self.exps[i] = [()] * self.n
            self.exps_pointer[i] = 0
        elif self.exps_pointer[i] == self.n - 1:
            # 如果没done，但是达到了临时经验池的长度，即n，则把最前边的经验存入， 并把之后的经验向前移动一位
            self.exps[i][0][-3:] = data[-3:]
            self.tree.add(self.max_p, self.exps[i][0])
            if self._size < self.capacity:
                self._size += 1
            del self.exps[i][0]
            self.exps[i].append(())
        else:
            # 如果没done，临时经验池也没满，就把指针后移
            self.exps_pointer[i] += 1
