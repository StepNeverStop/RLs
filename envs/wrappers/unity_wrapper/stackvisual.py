import numpy as np

from collections import deque
from envs.wrappers.unity_wrapper.wrappers import UnityReturnWrapper
from envs.wrappers.LazyFrames import LazyFrames

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
        self._stack_deque = {bn:deque([], maxlen=self._stack_nums) for bn in self.brain_names}

    def reset(self, **kwargs):
        self._env.reset(**kwargs)
        self._action_offset = {bn:0 for bn in self.brain_names}
        self._agent_ids = [self.list2dict(sr.agent_id) for sr in [self._env.get_step_result(bn) for bn in self.brain_names]]
        return self.get_reset_obs()

    def get_reset_obs(self):
        '''
        解析环境反馈的信息，将反馈信息分为四部分：向量、图像、奖励、done信号
        '''
        vector = []
        visual = []
        reward = []
        done = []
        for i, bn in enumerate(self.brain_names):
            step_result = self._env.get_step_result(bn)
            vec, vis, r, d = self.coordinate_reset_information(i, bn, step_result)
            vector.append(vec)
            visual.append(vis)
            reward.append(r)
            done.append(d)
        return zip(vector, visual, reward, done)
    
    def coordinate_reset_information(self, i, bn, sr):
        vector, visual, reward, done = super().coordinate_information(i, bn, sr)
        for _ in range(self._stack_nums):
            self._stack_deque[bn].append(visual)
        return (vector, np.concatenate(self._stack_deque[bn], axis=-1), reward, done)

    def coordinate_information(self, i, bn, sr):
        vector, visual, reward, done = super().coordinate_information(i, bn, sr)
        self._stack_deque[bn].append(visual)
        return (vector, np.concatenate(self._stack_deque[bn], axis=-1), reward, done)