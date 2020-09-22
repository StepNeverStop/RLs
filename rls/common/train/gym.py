#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from tqdm import trange
from copy import deepcopy
from typing import \
    Tuple, \
    List, \
    Callable, \
    NoReturn

from rls.utils.np_utils import \
    SMA, \
    arrprint
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)
bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'


def init_variables(env) -> Tuple[int, List[np.ndarray], List[np.ndarray]]:
    """
    inputs:
        env: Environment
    outputs:
        i: specify which item of state should be modified
        state: [vector_obs, visual_obs]
        newstate: [vector_obs, visual_obs]
    """
    i = 1 if env.obs_type == 'visual' else 0
    return (i,
            [np.full((env.n, 0), [], dtype=np.float32), np.full((env.n, 0), [], dtype=np.float32)],
            [np.full((env.n, 0), [], dtype=np.float32), np.full((env.n, 0), [], dtype=np.float32)])


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

    i, state, new_state = init_variables(env)
    sma = SMA(moving_average_episode)
    frame_step = begin_frame_step
    train_step = begin_train_step
    total_step = 0

    for episode in range(begin_episode, max_train_episode):
        model.reset()
        state[i] = env.reset()
        dones_flag = np.zeros(env.n)
        step = 0
        rets = np.zeros(env.n)
        last_done_step = -1
        while True:
            step += 1
            if render or episode > render_episode:
                env.render(record=False)
            action = model.choose_action(s=state[0], visual_s=state[1])
            new_state[i], reward, done, info, correct_new_state = env.step(action)
            rets += (1 - dones_flag) * reward
            dones_flag = np.sign(dones_flag + done)
            model.store_data(
                s=state[0],
                visual_s=state[1],
                a=action,
                r=reward,
                s_=new_state[0],
                visual_s_=new_state[1],
                done=done
            )
            model.partial_reset(done)
            state[i] = correct_new_state

            if policy_mode == 'off-policy':
                if total_step % off_policy_train_interval == 0:
                    model.learn(episode=episode, train_step=train_step)
                    train_step += 1
                if train_step % save_frequency == 0:
                    model.save_checkpoint(train_step=train_step, episode=episode, frame_step=frame_step)
                if off_policy_eval_interval > 0 and train_step % off_policy_eval_interval == 0:
                    gym_step_eval(deepcopy(env), model, train_step, off_policy_step_eval_episodes, max_step_per_episode)

            frame_step += env.n
            total_step += 1
            if 0 < max_train_step <= train_step or 0 < max_frame_step <= frame_step:
                model.save_checkpoint(train_step=train_step, episode=episode, frame_step=frame_step)
                logger.info(f'End Training, learn step: {train_step}, frame_step: {frame_step}')
                return

            if all(dones_flag):
                if last_done_step == -1:
                    last_done_step = step
                if policy_mode == 'off-policy':
                    break

            if step >= max_step_per_episode:
                break

        sma.update(rets)
        if policy_mode == 'on-policy':
            model.learn(episode=episode, train_step=train_step)
            train_step += 1
            if train_step % save_frequency == 0:
                model.save_checkpoint(train_step=train_step, episode=episode, frame_step=frame_step)
        model.writer_summary(
            episode,
            reward_mean=rets.mean(),
            reward_min=rets.min(),
            reward_max=rets.max(),
            step=last_done_step,
            **sma.rs
        )
        print_func(f'Eps: {episode:3d} | S: {step:4d} | LDS {last_done_step:4d} | R: {arrprint(rets, 2)}', out_time=True)

        if add_noise2buffer and episode % add_noise2buffer_episode_interval == 0:
            gym_no_op(env, model, pre_fill_steps=add_noise2buffer_steps, prefill_choose=False, desc='adding noise')

        if eval_while_train and env.reward_threshold is not None:
            if rets.max() >= env.reward_threshold:
                print_func(f'-------------------------------------------Evaluate episode: {episode:3d}--------------------------------------------------')
                gym_evaluate(env, model, max_step_per_episode, max_eval_episode, print_func)


def gym_step_eval(env, model,
                  step: int,
                  episodes_num: int,
                  max_step_per_episode: int) -> NoReturn:
    '''
    1个环境的推断模式
    '''
    cs = model.get_cell_state()  # 暂存训练时候的RNN隐状态

    i, state, _ = init_variables(env)
    sum_ret = 0.
    ave_steps = 0.
    for _ in trange(episodes_num, ncols=80, desc='evaluating', bar_format=bar_format):
        model.reset()
        state[i] = env.reset()
        ret = 0.
        step = 0
        while True:
            action = model.choose_action(s=state[0], visual_s=state[1], evaluation=True)
            _, reward, done, info, state[i] = env.step(action)
            model.partial_reset(done)
            reward = reward[0]
            done = done[0]
            ret += reward
            step += 1
            if done or step > max_step_per_episode:
                sum_ret += ret
                ave_steps += step
                break

    model.writer_summary(
        step,
        eval_return=sum_ret / episodes_num,
        eval_ave_step=ave_steps // episodes_num,
    )
    model.set_cell_state(cs)
    del env


def gym_evaluate(env, model,
                 max_step_per_episode: int,
                 max_eval_episode: int,
                 print_func: Callable[[str], None]) -> NoReturn:
    i, state, _ = init_variables(env)
    total_r = np.zeros(env.n)
    total_steps = np.zeros(env.n)

    for _ in trange(max_eval_episode, ncols=80, desc='evaluating', bar_format=bar_format):
        model.reset()
        state[i] = env.reset()
        dones_flag = np.zeros(env.n)
        steps = np.zeros(env.n)
        ret = np.zeros(env.n)
        while True:
            action = model.choose_action(s=state[0], visual_s=state[1], evaluation=True)  # In the future, this method can be combined with choose_action
            _, reward, done, info, state[i] = env.step(action)
            model.partial_reset(done)
            ret += (1 - dones_flag) * reward
            steps += (1 - dones_flag)
            dones_flag = np.sign(dones_flag + done)
            if all(dones_flag) or any(steps >= max_step_per_episode):
                break
        total_r += ret
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

    i, state, new_state = init_variables(env)
    model.reset()
    state[i] = env.reset()

    for _ in trange(0, pre_fill_steps, env.n, unit_scale=env.n, ncols=80, desc=desc, bar_format=bar_format):
        if prefill_choose:
            action = model.choose_action(s=state[0], visual_s=state[1])
        else:
            action = env.sample_actions()
        new_state[i], reward, done, info, correct_new_state = env.step(action)
        model.no_op_store(
            s=state[0],
            visual_s=state[1],
            a=action,
            r=reward,
            s_=new_state[0],
            visual_s_=new_state[1],
            done=done
        )
        model.partial_reset(done)
        state[i] = correct_new_state


def gym_inference(env, model,
                  episodes: int) -> NoReturn:
    i, state, _ = init_variables(env)
    for episode in range(episodes):
        step = 0
        model.reset()
        state[i] = env.reset()
        dones_flag = np.zeros(env.n)
        rets = np.zeros(env.n)
        while True:
            env.render(record=False)
            action = model.choose_action(s=state[0], visual_s=state[1], evaluation=True)
            step += 1
            _, reward, done, info, state[i] = env.step(action)
            rets += (1 - dones_flag) * reward
            dones_flag = np.sign(dones_flag + done)
            model.partial_reset(done)
            if dones_flag.all():
                logger.info(f'episode: {episode:4d}, returns: min {rets.min():6.2f}, mean {rets.mean():6.2f}, max {rets.max():6.2f}')
                break

            if step % 1000 == 0:
                logger.info(f'step: {step}')
