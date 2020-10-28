#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from collections import deque

from rls.envs.unity_wrapper.wrappers import UnityReturnWrapper
from rls.utils.LazyFrames import LazyFrames

# obs: [
#     brain1: [
#         agent1: [],
#         agent2: [],
#         ...
#         agentn: []
#     ],
#     brain2: [
#         agent1: [],
#         agent2: [],
#         ...
#         agentn: []
#     ],
#     ...
#     brainn: [
#         agent1: [],
#         agent2: [],
#         ...
#         agentn: []
#     ],
# ]

# =>

# obs: [
#     brain1: [
#         agent1: n*[],
#         agent2: n*[],
#         ...
#         agentn: n*[]
#     ],
#     brain2: [
#         agent1: n*[],
#         agent2: n*[],
#         ...
#         agentn: n*[]
#     ],
#     ...
#     brainn: [
#         agent1: n*[]
#         agent2: n*[]
#         ...
#         agentn: n*[]
#     ],
# ]


class StackVisualWrapper(UnityReturnWrapper):

    def __init__(self, env, stack_nums=4):
        super().__init__(env)
        self._stack_nums = stack_nums
        self._stack_deque = {gn: deque([], maxlen=self._stack_nums) for gn in self.group_names}

    def reset(self, **kwargs):
        self._env.reset(**kwargs)
        return self.get_reset_obs()

    def get_reset_obs(self):
        '''
        解析环境反馈的信息，将反馈信息分为四部分：向量、图像、奖励、done信号
        '''
        vector = []
        visual = []
        reward = []
        done = []
        info = []
        for i, gn in enumerate(self.group_names):
            vec, vis, r, d, ifo = self.coordinate_reset_information(i, gn)
            vector.append(vec)
            visual.append(vis)
            reward.append(r)
            done.append(d)
            info.append(ifo)
        return (vector, visual, reward, done, info)

    def coordinate_reset_information(self, i, gn):
        vector, visual, reward, done, info = super().coordinate_information(i, gn)
        for _ in range(self._stack_nums):
            self._stack_deque[gn].append(visual)
        return (vector, np.concatenate(self._stack_deque[gn], axis=-1), reward, done, info)

    def coordinate_information(self, i, gn):
        vector, visual, reward, done, info = super().coordinate_information(i, gn)
        self._stack_deque[gn].append(visual)
        return (vector, np.concatenate(self._stack_deque[gn], axis=-1), reward, done, info)
