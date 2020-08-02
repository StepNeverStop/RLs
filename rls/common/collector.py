#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from rls.distribute.utils.apex_utils import batch_numpy2proto
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


class GymCollector(object):

    @staticmethod
    def run_exps_stream(env, model):
        n = env.n
        i = 1 if env.obs_type == 'visual' else 0
        state = [np.array([[]] * n), np.array([[]] * n)]
        new_state = [np.array([[]] * n), np.array([[]] * n)]

        model.reset()
        state[i] = env.reset()
        dones_flag = np.zeros(n)

        while True:
            action = model.choose_action(s=state[0], visual_s=state[1])
            new_state[i], reward, done, info, correct_new_state = env.step(action)
            yield batch_numpy2proto([*state, action, reward, *new_state, done])
            model.partial_reset(done)
            state[i] = correct_new_state
            dones_flag = np.sign(dones_flag+done)
            if all(dones_flag):
                break

        logger.info('run_exps_stream success')

    @staticmethod
    def run_trajectory(env, model):
        n = env.n
        i = 1 if env.obs_type == 'visual' else 0
        state = [np.array([[]] * n), np.array([[]] * n)]
        new_state = [np.array([[]] * n), np.array([[]] * n)]
        trajectories = [[] for _ in range(n)]

        model.reset()
        state[i] = env.reset()
        dones_flag = np.zeros(n)

        while True:
            action = model.choose_action(s=state[0], visual_s=state[1])
            new_state[i], reward, done, info, correct_new_state = env.step(action)
            model.partial_reset(done)
            unfinished_index = np.where(dones_flag == 0)[0]
            for j in unfinished_index:
                trajectories[j].append(
                    (state[0][j], state[1][j], action[j], reward[j], new_state[0][j], new_state[1][j], done[j])
                )
            state[i] = correct_new_state
            dones_flag = np.sign(dones_flag+done)
            if all(dones_flag):
                break

        for traj in trajectories:
            yield batch_numpy2proto([np.asarray(arg) for arg in zip(*traj)])

        logger.info('run_trajectory success')


class UnityCollector(object):

    def __init__(self):
        pass

    @staticmethod
    def run_exps_stream(env, model, steps):
        pass
