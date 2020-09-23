#!/usr/bin/env python3
# encoding: utf-8

import ray
from enum import Enum
from typing import Dict


class OP(Enum):
    RESET = 0
    STEP = 1
    CLOSE = 2
    RENDER = 3
    SAMPLE = 4


@ray.remote
class RayEnv:
    def __init__(self, env_func, config: Dict):
        self.env = env_func(config)

    def seed(self, s):
        self.env.seed(s)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def sample(self):
        return self.env.action_sample()


def init_envs(func, config, n, seed):
    ray.init()
    envs = [RayEnv.remote(func, config) for i in range(n)]
    seeds = [seed + i for i in range(n)]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    [env.seed.remote(s) for env, s in zip(envs, seeds)]
    return envs


def op_func(envs, op: OP, _args=None):
    if op == OP.RESET:
        return ray.get([env.reset.remote() for env in envs])
    if op == OP.STEP:
        return ray.get([env.step.remote(action) for env, action in zip(envs, _args)])
    if op == OP.SAMPLE:
        return ray.get([env.sample.remote() for env in envs])
    if op == OP.RENDER:
        if _args:  # record
            [env.render.remote(filename=r'videos/{0}-{1}.mp4'.format(env.env.spec.id, i)) for i, env in enumerate(envs)]
        else:
            [env.render.remote() for env in envs]
        return None
    if op == OP.CLOSE:
        ray.get([env.close.remote() for env in envs])
        ray.shutdown()
        return None
