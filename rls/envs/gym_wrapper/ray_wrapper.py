#!/usr/bin/env python3
# encoding: utf-8

import ray

from typing import Dict


@ray.remote
class Env:
    def __init__(self, env_func, config: Dict):
        self.env = env_func(config)

    def seed(self, s):
        self.env.seed(s)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self, **kwargs):
        self.env.render(**kwargs)

    def close(self):
        self.env.close()

    def sample(self):
        return self.env.action_sample()


class RayEnv:

    def __init__(self, make_func, config, n, seed):
        ray.init()
        self.n = n
        self.idxs = list(range(n))
        self.envs = [Env.remote(make_func, (config, idx)) for idx in range(n)]
        for i in range(n):
            self.envs[i].seed.remote(seed + i)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def reset(self, idxs=[]):
        return ray.get([self.envs[i].reset.remote() for i in (idxs or self.idxs)])

    def step(self, actions, idxs=[]):
        return ray.get([self.envs[ray_idx].step.remote(actions[idx]) for idx, ray_idx in enumerate(idxs or self.idxs)])

    def render(self, record=False, idxs=[]):
        for i in (idxs or self.idxs):
            if record:
                self.envs[i].render.remote(filename=r'videos/{0}-{1}.mp4'.format(self.envs[i].env.spec.id, i))
            else:
                self.envs[i].render.remote()

    def close(self, idxs=[]):
        ray.get([self.envs[i].close.remote() for i in (idxs or self.idxs)])
        ray.shutdown()

    def sample(self, idxs=[]):
        return ray.get([self.envs[i].action_sample.remote() for i in (idxs or self.idxs)])
