import numpy as np
import tensorflow as tf
from .sum_tree import Sum_Tree
from abc import ABC, abstractmethod

# [s, visual_s, a, r, s_, visual_s_, done] must be this format.


class ReplayBuffer(ABC):
    def __init__(self, batch_size, capacity):
        assert isinstance(batch_size, int) and batch_size > 0, 'batch_size must be int and larger than 0'
        assert isinstance(capacity, int) and capacity > 0, 'capacity must be int and larger than 0'
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


class ExperienceReplay(ReplayBuffer):
    def __init__(self, batch_size, capacity):
        super().__init__(batch_size, capacity)
        self._data_pointer = 0
        self._buffer = np.empty(capacity, dtype=object)

    def add(self, *args):
        '''
        change [s, s],[a, a],[r, r] to [s, a, r],[s, a, r] and store every item in it.
        '''
        [self._store_op(data) for data in zip(*args)]

    def _store_op(self, data):
        self._buffer[self._data_pointer] = data
        self.update_rb_after_add()

    def sample(self):
        '''
        change [[s, a, r],[s, a, r]] to [[s, s],[a, a],[r, r]]
        '''
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        t = np.random.choice(self._buffer[:self._size], size=n_sample, replace=False)
        return [np.asarray(e) for e in zip(*t)]

    def get_all(self):
        return [np.asarray(e) for e in zip(*self._buffer[:self._size])]

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

    def __init__(self, batch_size, capacity, max_episode, alpha, beta, epsilon, global_v):
        '''
        inputs:
            max_episode: use for calculating the decay interval of beta
            alpha: control sampling rule, alpha -> 0 means uniform sampling, alpha -> 1 means complete td_error sampling
            beta: control importance sampling ratio, beta -> 0 means no IS, beta -> 1 means complete IS.
            epsilon: a small positive number that prevents td-error of 0 from never being replayed.
            global_v: whether using the global
        '''
        assert epsilon > 0, 'episode must larger than zero'
        super().__init__(batch_size, capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_interval = (1 - beta) / max_episode
        self.tree = Sum_Tree(capacity)
        self.epsilon = epsilon
        self.IS_w = 1   # weights of variables by using Importance Sampling
        self.min_p = 1
        self.max_p = epsilon
        self.global_v = global_v

    def add(self, *args):
        '''
        input: [ss, visual_ss, as, rs, s_s, visual_s_s, dones]
        '''
        self.add_batch(list(zip(*args)))
        # [self._store_op(data) for data in zip(*args)]

    def _store_op(self, data):
        self.tree.add(self.max_p, data)
        if self._size < self.capacity:
            self._size += 1

    def add_batch(self, data):
        data = list(data)
        num = len(data)
        self.tree.add_batch(np.full(num, self.max_p), data)
        self._size = min(self._size + num, self.capacity)

    def sample(self):
        '''
        output: weights, [ss, visual_ss, as, rs, s_s, visual_s_s, dones]
        '''
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        all_intervals = np.linspace(0, self.tree.total, n_sample)
        ps = np.random.uniform(all_intervals[:-1], all_intervals[1:])
        self.last_indexs, data_indx, p, data = self.tree.get_batch_parallel(ps)
        _min_p = self.min_p if self.global_v else p.min()
        self.IS_w = np.power(_min_p / p, self.beta)
        return data

    @property
    def is_lg_batch_size(self):
        return self._size > self.batch_size

    def update(self, priority, episode):
        '''
        input: priorities
        '''
        assert hasattr(priority, '__len__'), 'priority must have attribute of len()'
        assert len(priority) == len(self.last_indexs), 'length between priority and last_indexs must equal'
        self.beta += self.beta_interval * episode
        priority = np.power(np.abs(priority) + self.epsilon, self.alpha)
        self.min_p = min(self.min_p, priority.min())
        self.max_p = max(self.max_p, priority.max())
        self.tree._updatetree_batch(self.last_indexs, priority)
        # [self.tree._updatetree(idx, p) for idx, p in zip(self.last_indexs, priority)]

    def get_IS_w(self):
        return self.IS_w


class NStepWrapper:
    def __init__(self, buffer, gamma, n, agents_num):
        '''
        gamma: discount factor
        n: n step
        agents_num: batch experience
        '''
        self.buffer = buffer
        self.n = n
        self.gamma = gamma
        self.agents_num = agents_num
        self.queue = [[] for _ in range(agents_num)]

    def add(self, *args):
        '''
        change [s, s],[a, a],[r, r] to [s, a, r],[s, a, r] and store every item in it.
        '''
        [self._per_store(i, list(data)) for i, data in enumerate(zip(*args))]

    def _per_store(self, i, data):
        '''
        data:
            0   s           -7
            1   visual_s    -6
            2   a           -5
            3   r           -4
            4   s_          -3
            5   visual_s_   -2
            6   done        -1
        '''
        q = self.queue[i]
        if len(q) == 0:  # 如果Nstep临时经验池为空，就直接添加
            q.append(data)
            return
        if (q[-1][-3] != data[0]).any() or (q[-1][-2] != data[1]).any():    # 如果截断了，非常规done，把Nstep临时经验池中已存在的经验都存进去，临时经验池清空
            if len(q) == self.n:
                self._store_op(q.pop(0))
            else:
                q.clear()   # 保证经验池中不存在不足N长度的序列，有done的除外，因为（1-done）为0，导致gamma的次方计算不准确也没有关系。
            q.append(data)
            return

        if len(q) == self.n:  # 如果Nstep临时经验池满了，就把最早的一条经验存到经验池
            self._store_op(q.pop(0))
        _len = len(q)
        for j in range(_len):   # 然后再存入一条最新的经验到Nstep临时经验池
            q[j][3] += data[3] * (self.gamma ** (_len - j))
            q[j][4:] = data[4:]
        q.append(data)
        if data[-1]:  # done or not # 如果新数据是done，就清空临时经验池
            while q:    # (1-done)会清零不正确的n-step
                self._store_op(q.pop())

    def __getattr__(self, name):
        return getattr(self.buffer, name)


class NStepExperienceReplay(NStepWrapper):
    '''
    Replay Buffer + NStep
    [s, visual_s, a, r, s_, visual_s_, done] must be this format.
    '''

    def __init__(self, batch_size, capacity, gamma, n, agents_num):
        super().__init__(
            buffer=ExperienceReplay(batch_size, capacity),
            gamma=gamma, n=n, agents_num=agents_num
        )


class NStepPrioritizedExperienceReplay(NStepWrapper):
    '''
    PER + NStep
    [s, visual_s, a, r, s_, visual_s_, done] must be this format.
    '''

    def __init__(self, batch_size, capacity, max_episode, alpha, beta, epsilon, global_v, gamma, n, agents_num):
        super().__init__(
            buffer=PrioritizedExperienceReplay(batch_size, capacity, max_episode, alpha, beta, epsilon, global_v),
            gamma=gamma, n=n, agents_num=agents_num
        )


class EpisodeExperienceReplay(ReplayBuffer):
    
    def __init__(self, batch_size, capacity, agents_num):
        super().__init__(batch_size, capacity)
        self.agents_num = agents_num
        self.queue = [[] for _ in range(agents_num)]
        self._data_pointer = 0
        self._buffer = np.empty(capacity, dtype=object)

    def add(self, *args):
        '''
        change [s, s],[a, a],[r, r] to [s, a, r],[s, a, r] and store every item in it.
        '''
        [self._per_store(i, list(data)) for i, data in enumerate(zip(*args))]
    
    def _per_store(self, i, data):
        '''
        data:
            0   s           -7
            1   visual_s    -6
            2   a           -5
            3   r           -4
            4   s_          -3
            5   visual_s_   -2
            6   done        -1
        '''
        q = self.queue[i]
        if len(q) == 0:
            q.append(data)
            return
        if (q[-1][-3] != data[0]).any() or (q[-1][-2] != data[1]).any():
            self._store_op(q.copy())
            q.clear()
            q.append(data)
            return
        if data[-1]:
            q.append(data)
            self._store_op(q.copy())
            q.clear()
            return
        q.append(data)

    def _store_op(self, data):
        self._buffer[self._data_pointer] = data
        self.update_rb_after_add()

    def update_rb_after_add(self):
        self._data_pointer += 1
        if self._data_pointer >= self.capacity:  # replace when exceed the capacity
            self._data_pointer = 0
        if self._size < self.capacity:
            self._size += 1

    def sample(self):
        '''
        [B, (s, a, r, s', d)] => [B, time_step, N]
        '''
        n_sample = self.batch_size if self.is_lg_batch_size else self._size
        t = np.random.choice(self._buffer[:self._size], size=n_sample, replace=False)
        s, visual_s, a, r, s_, visual_s_, done = [[] for _ in range(7)]
        for i in t:
            data = [d for d in zip(*i)]
            s.append(data[0])
            visual_s.append(data[1])
            a.append(data[2])
            r.append(data[3])
            s_.append(data[4])
            visual_s_.append(data[5])
            done.append(data[6])
        # s, visual_s, a, r, s_, visual_s_, done = map(np.asarray, [s, visual_s, a, r, s_, visual_s_, done])
        pad = lambda x: tf.keras.preprocessing.sequence.pad_sequences(x, padding='post', dtype='float32', value=0.)
        s, visual_s, a, r, s_, visual_s_ = map(pad, [s, visual_s, a, r, s_, visual_s_])
        done = tf.keras.preprocessing.sequence.pad_sequences(done, padding='post', dtype='float32', value=1.)
        # a, r, done = map(lambda x: tf.reshape(x, (-1, x.shape[-1])), [a, r, done])  # [B, T, N] => [B*T, N]
        s, visual_s, a, r, s_, visual_s_, done = map(lambda x: tf.reshape(x, (-1, x.shape[-1])), [s, visual_s, a, r, s_, visual_s_, done])  # [B, T, N] => [B*T, N]
        return (s, visual_s, a, r, s_, visual_s_, done)

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

