#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from typing import (List,
                    Callable)


def multi_agents_data_preprocess(copy_nums: int, group_controls: List) -> Callable[..., List]:
    '''
    处理多个环境副本下的多智能体训练数据, 将数据格式从
    [组1:[组1控制的智能体数量, 组1的维度], 组2:[组2控制的智能体数量, 组2的维度], ...]
    转换为
    [智能体1:[环境副本数量, 智能体1的维度], 智能体2:[环境副本数量, 智能体2的维度], ...]

    这么做的目的是方便多智能体环境下的多环境训练数据批量存储
    '''

    def data_change_func(data):
        # Copys: Batch
        # [Brains, Agents, Dims] => [Brains, Copys, Agents_perCopy_perBrain, Dims]
        data = [np.asarray(x).reshape(copy_nums, group_controls[i], -1) for i, x in enumerate(data)]
        # [Brains, Copys, Agents_perCopy_perBrain, Dims] => [Agents_perCopy_perBrain * Brains, Copys, Dims]: [Agents_perCopy, Copys, Dims]
        l = []
        for d in data:  # brains
            for i in range(d.shape[1]):  # agents
                l.append(d[:, i])   # 总共append了 sum([brain1_controls_percopy, brain2_controls_percopy, ...]) 次
        return l    # [total_agents, batch, dimension]

    return data_change_func


def multi_agents_action_reshape(copy_nums: int, group_controls: List) -> Callable[..., List]:

    def action_reshape_func(actions):
        '''
        actions : [Agents_perCopy_perBrain * Brains, Copys, Dims]
        '''

        l = []
        start = 0
        for i in group_controls:
            l.append(actions[start:start + i])
            start += i
        # l: [Brains, Agents_perCopy_perBrain, Copys, Dims] => [Brains, Agents, Dims]
        return [np.asarray(x).reshape(group_controls[i] * copy_nums, -1) for i, x in enumerate(l)]

    return action_reshape_func
