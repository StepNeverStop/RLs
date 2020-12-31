#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from tqdm import trange
from typing import (Callable,
                    NoReturn)

from rls.utils.np_utils import (SMA,
                                arrprint)
from rls.utils.mlagents_utils import (multi_agents_data_preprocess,
                                      multi_agents_action_reshape)
from rls.utils.logging_utils import get_logger

logger = get_logger(__name__)
bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'


def unity_train(env, model,
                print_func: Callable[[str], None],
                begin_train_step: int,
                begin_frame_step: int,
                begin_episode: int,
                save_frequency: int,
                max_step_per_episode: int,
                max_train_episode: int,
                policy_mode: str,
                moving_average_episode: int,
                add_noise2buffer: bool,
                add_noise2buffer_episode_interval: int,
                add_noise2buffer_steps: int,
                max_train_step: int,
                max_frame_step: int,
                real_done: bool,
                off_policy_train_interval: int) -> NoReturn:
    """
    TODO: Annotation
    Train loop. Execute until episode reaches its maximum or press 'ctrl+c' artificially.
    Inputs:
        env:                    Environment for interaction.
        model:                  all model for this training task.
        save_frequency:         how often to save checkpoints.
        max_step_per_episode:   maximum number of steps for an episode.
        resampling_interval:    how often to resample parameters for env reset.
    """

    sma = SMA(moving_average_episode)
    frame_step = begin_frame_step
    train_step = begin_train_step
    n = env.behavior_agents[env.first_bn]

    for episode in range(begin_episode, max_train_episode):
        model.reset()
        ret = env.reset()[env.first_bn]
        s = ret.corrected_vector
        visual_s = ret.corrected_visual
        dones_flag = np.zeros(n, dtype=float)
        rewards = np.zeros(n, dtype=float)
        step = 0
        last_done_step = -1

        while True:
            step += 1
            action = model.choose_action(s=s, visual_s=visual_s)
            ret = env.step({env.first_bn: action})[env.first_bn]

            model.store_data(
                s=s,
                visual_s=visual_s,
                a=action,
                r=ret.reward,
                s_=ret.vector,
                visual_s_=ret.visual,
                done=ret.info['real_done'] if real_done else ret.done
            )
            model.partial_reset(ret.done)
            rewards += (1 - dones_flag) * ret.reward
            dones_flag = np.sign(dones_flag + ret.done)
            s = ret.corrected_vector
            visual_s = ret.corrected_visual

            if policy_mode == 'off-policy':
                if train_step % off_policy_train_interval == 0:
                    model.learn(episode=episode, train_step=train_step)
                train_step += 1
                if train_step % save_frequency == 0:
                    model.save_checkpoint(train_step=train_step, episode=episode, frame_step=frame_step)

            frame_step += n
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

        sma.update(rewards)
        if policy_mode == 'on-policy':
            model.learn(episode=episode, train_step=train_step)
            train_step += 1
            if train_step % save_frequency == 0:
                model.save_checkpoint(train_step=train_step, episode=episode, frame_step=frame_step)
        model.writer_summary(
            episode,
            reward_mean=rewards.mean(),
            reward_min=rewards.min(),
            reward_max=rewards.max(),
            step=last_done_step,
            **sma.rs
        )
        print_func(f'Eps {episode:3d} | S {step:4d} | LDS {last_done_step:4d}', out_time=True)
        print_func(f'{env.first_bn} R: {arrprint(rewards, 2)}')

        if add_noise2buffer and episode % add_noise2buffer_episode_interval == 0:
            unity_no_op(env, model, pre_fill_steps=add_noise2buffer_steps, prefill_choose=False, real_done=real_done,
                        desc='adding noise')


def unity_no_op(env, model,
                pre_fill_steps: int,
                prefill_choose: bool,
                real_done: bool,
                desc: str = 'Pre-filling') -> NoReturn:
    '''
    Interact with the environment but do not perform actions. Prepopulate the ReplayBuffer.
    Make sure steps is greater than n-step if using any n-step ReplayBuffer.
    '''
    assert isinstance(pre_fill_steps, int) and pre_fill_steps >= 0, 'no_op.steps must have type of int and larger than/equal 0'
    n = env.behavior_agents[env.first_bn]

    if pre_fill_steps == 0:
        return
    model.reset()
    ret = env.reset()[env.first_bn]
    s = ret.corrected_vector
    visual_s = ret.corrected_visual

    for _ in trange(0, pre_fill_steps, n, unit_scale=n, ncols=80, desc=desc, bar_format=bar_format):
        if prefill_choose:
            action = model.choose_action(s=s, visual_s=visual_s)
        else:
            action = env.random_action()[env.first_bn]
        ret = env.step({env.first_bn: action})[env.first_bn]
        model.no_op_store(
            s=s,
            visual_s=visual_s,
            a=action,
            r=ret.reward,
            s_=ret.vector,
            visual_s_=ret.visual,
            done=ret.info['real_done'] if real_done else ret.done
        )
        model.partial_reset(ret.done)
        s = ret.corrected_vector
        visual_s = ret.corrected_visual


def unity_inference(env, model,
                    episodes: int) -> NoReturn:
    """
    inference mode. algorithm model will not be train, only used to show agents' behavior
    """

    for episode in range(episodes):
        model.reset()
        ret = env.reset()[env.first_bn]
        while True:
            action = model.choose_action(s=ret.corrected_vector,
                                         visual_s=ret.corrected_visual,
                                         evaluation=True)
            model.partial_reset(ret.done)
            ret = env.step({env.first_bn: action})[env.first_bn]


def ma_unity_no_op(env, model,
                   pre_fill_steps: int,
                   prefill_choose: bool,
                   desc: str = 'Pre-filling',
                   real_done: bool = True) -> NoReturn:
    assert isinstance(pre_fill_steps, int) and pre_fill_steps >= 0, 'multi-agent no_op.steps must have type of int'

    if pre_fill_steps == 0:
        return

    data_change_func = multi_agents_data_preprocess(env.env_copys, env.behavior_controls)
    action_reshape_func = multi_agents_action_reshape(env.env_copys, env.behavior_controls)
    model.reset()

    # [s(s_brain1(agent1, agent2, ...), s_brain2, ...), visual_s, r, done, info]
    s, visual_s, _, _, _, _, _ = env.reset()
    # [total_agents, batch, dimension]
    s, visual_s = map(data_change_func, [s, visual_s])

    for _ in trange(0, pre_fill_steps, env.env_copys, unit_scale=env.env_copys, ncols=80, desc=desc, bar_format=bar_format):
        if prefill_choose:
            action = model.choose_action(s=s, visual_s=visual_s)    # [total_agents, batch, dimension]
            action = action_reshape_func(action)
            actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.behavior_names)}
        else:
            actions = env.random_action()
            action = list(actions.values())
        s_, visual_s_, r, done, info, corrected_s_, corrected_visual_s_ = env.step(actions)
        if real_done:
            done = [g['real_done'] for g in info]

        action, r, done, s_, visual_s_, corrected_s_, corrected_visual_s_ = map(data_change_func, [action, r, done, s_, visual_s_, corrected_s_, corrected_visual_s_])
        done = np.asarray(done).sum((0, 2))

        model.no_op_store(
            *s,
            *visual_s,
            *action,
            *r,
            *s_,
            *visual_s_,
            done[np.newaxis, :]
        )
        model.partial_reset(done)
        s = corrected_s_
        visual_s = corrected_visual_s_


def ma_unity_train(env, model,
                   print_func: Callable[[str], None],
                   begin_train_step: int,
                   begin_frame_step: int,
                   begin_episode: int,
                   max_train_step: int,
                   max_frame_step: int,
                   off_policy_train_interval: int,
                   moving_average_episode: int,
                   save_frequency: int,
                   max_step_per_episode: int,
                   max_train_episode: int,
                   policy_mode: str,
                   real_done: bool = True) -> NoReturn:
    assert policy_mode == 'off-policy', "multi-agents algorithms now support off-policy only."

    frame_step = begin_frame_step
    train_step = begin_train_step

    data_change_func = multi_agents_data_preprocess(env.env_copys, env.behavior_controls)
    action_reshape_func = multi_agents_action_reshape(env.env_copys, env.behavior_controls)
    agents_num_per_copy = sum(env.behavior_controls)

    sma = [SMA(moving_average_episode) for _ in range(agents_num_per_copy)]

    for episode in range(begin_episode, max_train_episode):

        dones_flag = np.zeros(env.env_copys)
        rewards = np.zeros((agents_num_per_copy, env.env_copys))

        model.reset()
        s, visual_s, _, _, _, _, _ = env.reset()
        s, visual_s = map(data_change_func, [s, visual_s])

        step = 0
        last_done_step = -1
        while True:
            action = model.choose_action(s=s, visual_s=visual_s)    # [total_agents, batch, dimension]
            action = action_reshape_func(action)
            actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.behavior_names)}
            s_, visual_s_, r, done, info, corrected_s_, corrected_visual_s_ = env.step(actions)    # [Brains, Agents, Dims]
            step += 1

            if real_done:
                done = [g['real_done'] for g in info]

            # [Agents_perCopy, Copys, Dims]
            action, r, done, s_, visual_s_, corrected_s_, corrected_visual_s_ = map(data_change_func, [action, r, done, s_, visual_s_, corrected_s_, corrected_visual_s_])
            done = np.sign(np.asarray(done).sum((0, 2)))  # [Copys,]

            rewards += np.asarray(r).reshape(-1, env.env_copys) * (1 - dones_flag)

            dones_flag = np.sign(dones_flag + done)
            model.store_data(
                *s,
                *visual_s,
                *action,
                *r,
                *s_,
                *visual_s_,
                done[np.newaxis, :]
            )
            model.partial_reset(done)
            s = corrected_s_
            visual_s = corrected_visual_s_

            if policy_mode == 'off-policy':
                if train_step % off_policy_train_interval == 0:
                    model.learn(episode=episode, train_step=train_step)
                train_step += 1
                if train_step % save_frequency == 0:
                    model.save_checkpoint(train_step=train_step, episode=episode, frame_step=frame_step)

            frame_step += 1
            if 0 < max_train_step < train_step or 0 < max_frame_step < frame_step:
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

        for i in range(agents_num_per_copy):
            sma[i].update(rewards[i])
            model.writer_summary(
                episode,
                agent_idx=i,
                reward_mean=rewards[i].mean(),
                reward_min=rewards[i].min(),
                reward_max=rewards[i].max(),
                # step=last_done_step,
                **sma[i].rs
            )

        print_func(f'Eps {episode:3d} | S {step:4d} | LDS {last_done_step:4d}', out_time=True)
        for i in range(agents_num_per_copy):
            print_func(f'Agent {i} R: {arrprint(rewards[i], 2)}')


def ma_unity_inference(env, model,
                       episodes: int) -> NoReturn:
    """
    inference mode. algorithm model will not be train, only used to show agents' behavior
    """
    data_change_func = multi_agents_data_preprocess(env.env_copys, env.behavior_controls)
    action_reshape_func = multi_agents_action_reshape(env.env_copys, env.behavior_controls)
    for episode in range(episodes):
        model.reset()
        s, visual_s, _, _, _, _, _ = env.reset()
        while True:
            action = model.choose_action(s=s, visual_s=visual_s, evaluation=True)    # [total_agents, batch, dimension]
            action = action_reshape_func(action)
            actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.behavior_names)}
            _, _, _, _, _, s, visual_s_ = env.step(actions)
