#!/usr/bin/env python3
# encoding: utf-8

import multiprocessing

from enum import Enum
from typing import Dict


class OP(Enum):
    RESET = 0
    STEP = 1
    CLOSE = 2
    RENDER = 3
    SAMPLE = 4
    SEED = 5


class MultiProcessingEnv:
    def __init__(self, make_func, config: Dict, n, seed):
        self.n = n
        self.idxs = list(range(n))
        self.parent_conns = []
        for i in range(n):
            parent_conn, child_conn = multiprocessing.Pipe()
            multiprocessing.Process(target=MultiProcessingEnv.run,
                                    args=(i,
                                          make_func,
                                          config,
                                          child_conn),
                                    daemon=True).start()
            self.parent_conns.append(parent_conn)

        self.seed(seeds=[seed + i for i in range(n)])  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    @staticmethod
    def run(i, make_func, config, conn):
        env = make_func(config, i)
        while True:
            op, data = conn.recv()
            if op == OP.SEED:
                env.seed(data)
            elif op == OP.RESET:
                conn.send(env.reset())
            elif op == OP.STEP:
                conn.send(env.step(data))
            elif op == OP.RENDER:
                if data:
                    env.render(filename=r'videos/{0}-{1}.mp4'.format(env.env.spec.id, i))
                else:
                    env.render()
            elif op == OP.SAMPLE:
                conn.send(env.action_sample())
            elif op == OP.CLOSE:
                env.close()
                break

    def seed(self, seeds, idxs=[]):
        for idx, conn_idx in enumerate(idxs or self.idxs):
            self.parent_conns[conn_idx].send((OP.SEED, seeds[idx]))

    def reset(self, idxs=[]):
        ret = []
        for i in (idxs or self.idxs):
            self.parent_conns[i].send((OP.RESET, None))
            ret.append(self.parent_conns[i].recv())
        return ret

    def step(self, actions, idxs=[]):
        ret = []
        for idx, conn_idx in enumerate(idxs or self.idxs):
            self.parent_conns[conn_idx].send((OP.STEP, actions[idx]))
            ret.append(self.parent_conns[conn_idx].recv())
        return ret

    def render(self, record=False, idxs=[]):
        for i in (idxs or self.idxs):
            self.parent_conns[i].send((OP.RENDER, record))

    def close(self, idxs=[]):
        for i in (idxs or self.idxs):
            self.parent_conns[i].send((OP.CLOSE, None))

    def sample(self, idxs=[]):
        ret = []
        for i in (idxs or self.idxs):
            self.parent_conns[i].send((OP.SAMPLE, None))
            ret.append(self.parent_conns[i].recv())
        return ret
