#!/usr/bin/env python3
# encoding: utf-8

class VectorEnv:

    def __init__(self, make_func, config, n, seed):
        self.n = n
        self.idxs = list(range(n))
        self.envs = [make_func(idx, **config) for idx in range(n)]
        for i in range(n):
            self.envs[i].seed(seed + i)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def reset(self, idxs=[]):
        return [self.envs[i].reset() for i in (idxs or self.idxs)]

    def step(self, actions, idxs=[]):
        return [self.envs[env_idx].step(actions[idx]) for idx, env_idx in enumerate(idxs or self.idxs)]

    def render(self, record=False, idxs=[]):
        for i in (idxs or self.idxs):
            if record:
                self.envs[i].render(filename=r'videos/{0}-{1}.mp4'.format(self.envs[i].env.spec.id, i))
            else:
                self.envs[i].render()

    def close(self, idxs=[]):
        [self.envs[i].close() for i in (idxs or self.idxs)]

    def action_sample(self, idxs=[]):
        return [self.envs[i].action_sample() for i in (idxs or self.idxs)]
