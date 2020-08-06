#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import numpy as np

from copy import deepcopy

from rls.utils.np_utils import \
    SMA, \
    arrprint
from rls.distribute.utils.apex_utils import \
    batch_numpy2proto, \
    exps_and_tderror2proto
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


class GymCollector(object):

    @staticmethod
    def evaluate(env, model):
        n = env.n
        i = 1 if env.obs_type == 'visual' else 0
        state = [np.full((n, 0), []), np.full((n, 0), [])]
        sma = SMA(100)
        total_step = 0
        episode = 0

        while True:
            episode += 1
            model.reset()
            state[i] = env.reset()
            dones_flag = np.zeros(env.n)
            step = 0
            rets = np.zeros(env.n)
            last_done_step = -1
            while True:
                step += 1
                # env.render(record=False)
                action = model.choose_action(s=state[0], visual_s=state[1])
                _, reward, done, info, state[i] = env.step(action)
                rets += (1 - dones_flag) * reward
                dones_flag = np.sign(dones_flag + done)
                model.partial_reset(done)
                total_step += 1
                if all(dones_flag):
                    if last_done_step == -1:
                        last_done_step = step
                    break

                if step >= 200:
                    break

            sma.update(rets)
            model.writer_summary(
                episode,
                reward_mean=rets.mean(),
                reward_min=rets.min(),
                reward_max=rets.max(),
                step=last_done_step,
                **sma.rs
            )
            print(f'Eps: {episode:3d} | S: {step:4d} | LDS {last_done_step:4d} | R: {arrprint(rets, 2)}')
            time.sleep(5)

    @staticmethod
    def run_exps_stream(env, model):
        n = env.n
        i = 1 if env.obs_type == 'visual' else 0
        state = [np.full((n, 0), []), np.full((n, 0), [])]
        new_state = [np.full((n, 0), []), np.full((n, 0), [])]

        model.reset()
        state[i] = env.reset()
        dones_flag = np.zeros(n)

        while True:
            action = model.choose_action(s=state[0], visual_s=state[1], evaluation=True)
            new_state[i], reward, done, info, correct_new_state = env.step(action)
            exps = [*state, action, reward[:, np.newaxis], *new_state, done[:, np.newaxis]]
            td_error = model.apex_cal_td(deepcopy(exps))
            yield exps_and_tderror2proto(
                exps=exps,
                td_error=td_error)
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
        state = [np.full((n, 0), []), np.full((n, 0), [])]
        new_state = [np.full((n, 0), []), np.full((n, 0), [])]
        trajectories = [[] for _ in range(n)]

        model.reset()
        state[i] = env.reset()
        dones_flag = np.zeros(n)

        while True:
            action = model.choose_action(s=state[0], visual_s=state[1], evaluation=True)
            new_state[i], reward, done, info, correct_new_state = env.step(action)
            model.partial_reset(done)
            unfinished_index = np.where(dones_flag == 0)[0]
            for j in unfinished_index:
                trajectories[j].append(
                    (state[0][j], state[1][j], action[j], reward[:, np.newaxis][j], new_state[0][j], new_state[1][j], done[:, np.newaxis][j])
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
