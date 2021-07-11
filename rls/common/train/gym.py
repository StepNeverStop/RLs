#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from tqdm import trange
from copy import deepcopy
from typing import (Tuple,
                    List,
                    Callable,
                    NoReturn)

from rls.common.recoder import SimpleMovingAverageRecoder
from rls.common.specs import BatchExperiences
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)
bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'


def gym_train(env, model,
              print_func: Callable[[str], None],
              begin_train_step: int,
              begin_frame_step: int,
              begin_episode: int,
              render: bool,
              render_episode: int,
              save_frequency: int,
              max_step_per_episode: int,
              max_train_episode: int,
              eval_while_train: bool,
              max_eval_episode: int,
              off_policy_step_eval_episodes: int,
              off_policy_train_interval: int,
              policy_mode: str,
              moving_average_episode: int,
              add_noise2buffer: bool,
              add_noise2buffer_episode_interval: int,
              add_noise2buffer_steps: int,
              off_policy_eval_interval: int,
              max_train_step: int,
              max_frame_step: int) -> NoReturn:
    """
    TODO: Annotation
    """

    recoder = SimpleMovingAverageRecoder(n_copys=env.n, gamma=0.99, verbose=True,
                                         length=moving_average_episode)
    frame_step = begin_frame_step
    train_step = begin_train_step

    for episode in range(begin_episode, max_train_episode):
        model.reset()
        obs = env.reset()
        recoder.episode_reset(episode=episode)
        for _ in range(max_step_per_episode):
            if render or episode > render_episode:
                env.render(record=False)
            action = model(obs=obs)
            ret = env.step(action)
            model.store_data(BatchExperiences(obs=obs,
                                              action=action,
                                              reward=ret.reward[:, np.newaxis],  # [B, ] => [B, 1]
                                              obs_=ret.obs,
                                              done=ret.done[:, np.newaxis]))
            model.partial_reset(ret.done)
            recoder.step_update(rewards=ret.reward, dones=ret.done)
            obs = ret.corrected_obs

            if policy_mode == 'off-policy':
                if recoder.total_step % off_policy_train_interval == 0:
                    model.learn(episode=episode, train_step=train_step)
                    train_step += 1
                if train_step % save_frequency == 0:
                    model.save(train_step=train_step, episode=episode, frame_step=frame_step)
                if off_policy_eval_interval > 0 and train_step % off_policy_eval_interval == 0:
                    gym_step_eval(deepcopy(env), model, train_step, off_policy_step_eval_episodes, max_step_per_episode)

            frame_step += env.n
            if 0 < max_train_step <= train_step or 0 < max_frame_step <= frame_step:
                model.save(train_step=train_step, episode=episode, frame_step=frame_step)
                logger.info(f'End Training, learn step: {train_step}, frame_step: {frame_step}')
                return

            if recoder.is_all_done and policy_mode == 'off-policy':
                break

        recoder.episode_end()
        if policy_mode == 'on-policy':
            model.learn(episode=episode, train_step=train_step)
            train_step += 1
            if train_step % save_frequency == 0:
                model.save(train_step=train_step, episode=episode, frame_step=frame_step)
        model.writer_summary(episode, recoder.summary_dict)
        print_func(str(recoder), out_time=True)

        if add_noise2buffer and episode % add_noise2buffer_episode_interval == 0:
            gym_no_op(env, model, pre_fill_steps=add_noise2buffer_steps, prefill_choose=False, desc='adding noise')

        if eval_while_train and env.reward_threshold is not None:
            if recoder.total_returns.max() >= env.reward_threshold:
                print_func(f'-------------------------------------------Evaluate episode: {episode:3d}--------------------------------------------------')
                gym_evaluate(env, model, max_step_per_episode, max_eval_episode, print_func)


def gym_step_eval(env, model,
                  train_step: int,
                  episodes_num: int,
                  max_step_per_episode: int) -> NoReturn:
    '''
    1个环境的推断模式
    '''
    cs = model.get_cell_state()  # 暂存训练时候的RNN隐状态

    sum_ret = 0.
    ave_steps = 0.
    for _ in trange(episodes_num, ncols=80, desc='evaluating', bar_format=bar_format):
        model.reset()
        obs = env.reset()
        returns = 0.
        step = 0
        for _ in range(max_step_per_episode):
            action = model(obs=obs, evaluation=True)
            ret = env.step(action)
            model.partial_reset(ret.done)
            returns += ret.reward[0]
            step += 1
            if ret.done[0]:
                break
            obs = ret.corrected_obs
        sum_ret += returns
        ave_steps += step

    model.writer_summary(
        train_step,
        dict(
            eval_return=sum_ret / episodes_num,
            eval_ave_step=ave_steps // episodes_num
        )
    )
    model.set_cell_state(cs)
    del env


def gym_evaluate(env, model,
                 max_step_per_episode: int,
                 max_eval_episode: int,
                 print_func: Callable[[str], None]) -> NoReturn:
    total_r = np.zeros(env.n)
    total_steps = np.zeros(env.n)

    for _ in trange(max_eval_episode, ncols=80, desc='evaluating', bar_format=bar_format):
        model.reset()
        obs = env.reset()
        dones_flag = np.zeros(env.n)
        steps = np.zeros(env.n)
        returns = np.zeros(env.n)
        for _ in range(max_step_per_episode):
            action = model(obs=obs, evaluation=True)
            ret = env.step(action)
            model.partial_reset(ret.done)
            returns += (1 - dones_flag) * ret.reward
            steps += (1 - dones_flag)
            dones_flag = np.sign(dones_flag + ret.done)
            if all(dones_flag):
                break
            obs = ret.corrected_obs
        total_r += returns
        total_steps += steps
    average_r = total_r.mean() / max_eval_episode
    average_step = int(total_steps.mean() / max_eval_episode)
    solved = True if average_r >= env.reward_threshold else False
    print_func(f'evaluate number: {max_eval_episode:3d} | average step: {average_step} | average reward: {average_r} | SOLVED: {solved}')
    print_func('----------------------------------------------------------------------------------------------------------------------------')


def gym_no_op(env, model,
              pre_fill_steps: int,
              prefill_choose: bool,
              desc: str = 'Pre-filling') -> NoReturn:
    assert isinstance(pre_fill_steps, int) and pre_fill_steps >= 0, 'no_op.steps must have type of int and larger than/equal 0'

    if pre_fill_steps == 0:
        return

    model.reset()
    obs = env.reset()

    for _ in trange(0, pre_fill_steps, env.n, unit_scale=env.n, ncols=80, desc=desc, bar_format=bar_format):
        if prefill_choose:
            action = model(obs=obs)
        else:
            action = env.sample_actions()
        ret = env.step(action)
        model.no_op_store(BatchExperiences(obs=obs,
                                           action=action,
                                           reward=ret.reward[:, np.newaxis],  # [B, ] => [B, 1]
                                           obs_=ret.obs,
                                           done=ret.done[:, np.newaxis]))
        model.partial_reset(ret.done)
        obs = ret.corrected_obs


def gym_inference(env, model, episodes: int) -> NoReturn:
    for episode in range(episodes):
        step = 0
        model.reset()
        obs = env.reset()
        dones_flag = np.zeros(env.n)
        returns = np.zeros(env.n)
        while True:
            env.render(record=False)
            action = model(obs=obs, evaluation=True)
            step += 1
            ret = env.step(action)
            returns += (1 - dones_flag) * ret.reward
            dones_flag = np.sign(dones_flag + ret.done)
            model.partial_reset(ret.done)
            obs = ret.corrected_obs
            if dones_flag.all():
                logger.info(f'episode: {episode:4d}, returns: min {returns.min():6.2f}, mean {returns.mean():6.2f}, max {returns.max():6.2f}')
                break

            if step % 1000 == 0:
                logger.info(f'step: {step}')
