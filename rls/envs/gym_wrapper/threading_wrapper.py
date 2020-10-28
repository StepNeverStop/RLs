#!/usr/bin/env python3
# encoding: utf-8

import threading


class FakeMultiThread(threading.Thread):

    def __init__(self, make_func, args=()):
        super().__init__()
        self.make_func = make_func
        self.args = args

    def run(self):
        self.result = self.make_func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


class MultiThreadEnv:

    def __init__(self, make_func, config, n, seed):
        self.n = n
        self.idxs = list(range(n))
        self.envs = [make_func(config, idx) for idx in range(n)]
        for i in range(n):
            self.envs[i].seed(seed + i)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def reset(self, idxs=[]):
        threadpool = []
        for i in (idxs or self.idxs):
            threadpool.append(FakeMultiThread(self.envs[i].reset, args=()))
        for th in threadpool:
            th.start()
        for th in threadpool:
            th.join()
        return [th.get_result() for th in threadpool]

    def step(self, actions, idxs=[]):
        threadpool = []
        for idx, th_idx in enumerate(idxs or self.idxs):
            threadpool.append(FakeMultiThread(self.envs[th_idx].step, args=(actions[idx],)))
        for th in threadpool:
            th.start()
        for th in threadpool:
            th.join()
        return [th.get_result() for th in threadpool]

    def render(self, record=False, idxs=[]):
        for i in (idxs or self.idxs):
            if record:
                self.envs[i].render(filename=r'videos/{0}-{1}.mp4'.format(self.envs[i].env.spec.id, i))
            else:
                self.envs[i].render()

    def close(self, idxs=[]):
        threadpool = []
        for i in (idxs or self.idxs):
            threadpool.append(FakeMultiThread(self.envs[i].close, args=()))
        for th in threadpool:
            th.start()
        for th in threadpool:
            th.join()

    def sample(self, idxs=[]):
        threadpool = []
        for i in (idxs or self.idxs):
            threadpool.append(FakeMultiThread(self.envs[i].action_sample, args=()))
        for th in threadpool:
            th.start()
        for th in threadpool:
            th.join()
        return [th.get_result() for th in threadpool]
