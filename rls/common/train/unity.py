#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import numpy as np

from tqdm import trange

from rls.utils.np_utils import \
    SMA, \
    arrprint
from rls.utils.list_utils import zeros_initializer
from rls.utils.mlagents_utils import \
    multi_agents_data_preprocess, \
    multi_agents_action_reshape

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("common.train.unity")
bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'


def unity_train(env, models, print_func,
                begin_train_step, begin_frame_step, begin_episode, save_frequency, max_step_per_episode, max_train_episode, policy_mode,
                moving_average_episode, add_noise2buffer, add_noise2buffer_episode_interval, add_noise2buffer_steps,
                max_train_step, max_frame_step, real_done, off_policy_train_interval):
    """
    TODO: Annotation
    Train loop. Execute until episode reaches its maximum or press 'ctrl+c' artificially.
    Inputs:
        env:                    Environment for interaction.
        models:                 all models for this training task.
        save_frequency:         how often to save checkpoints.
        max_step_per_episode:   maximum number of steps for an episode.
        resampling_interval:    how often to resample parameters for env reset.
    Variables:
        brain_names:    a list of brain names set in Unity.
        state: store    a list of states for each brain. each item contain a list of states for each agents that controlled by the same brain.
        visual_state:   store a list of visual state information for each brain.
        action:         store a list of actions for each brain.
        dones_flag:     store a list of 'done' for each brain. use for judge whether an episode is finished for every agents.
        rewards:        use to record rewards of agents for each brain.
    """

    state, visual_state, action, dones_flag, rewards = zeros_initializer(env.brain_num, 5)
    sma = [SMA(moving_average_episode) for i in range(env.brain_num)]
    frame_step = begin_frame_step
    min_of_all_agents = min(env.brain_agents)
    train_step = [begin_train_step for _ in range(env.brain_num)]

    for episode in range(begin_episode, max_train_episode):
        [model.reset() for model in models]
        ObsRewDone = zip(*env.reset())
        for i, (_v, _vs, _r, _d, _info) in enumerate(ObsRewDone):
            dones_flag[i] = np.zeros(env.brain_agents[i])
            rewards[i] = np.zeros(env.brain_agents[i])
            state[i] = _v
            visual_state[i] = _vs
        step = 0
        last_done_step = -1
        while True:
            step += 1
            for i in range(env.brain_num):
                action[i] = models[i].choose_action(s=state[i], visual_s=visual_state[i])
            actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.brain_names)}
            ObsRewDone = zip(*env.step(actions))

            for i, (_v, _vs, _r, _d, _info) in enumerate(ObsRewDone):
                models[i].store_data(
                    s=state[i],
                    visual_s=visual_state[i],
                    a=action[i],
                    r=_r,
                    s_=_v,
                    visual_s_=_vs,
                    done=_info['real_done'] if real_done else _d
                )
                models[i].partial_reset(_d)
                rewards[i] += (1 - dones_flag[i]) * _r
                dones_flag[i] = np.sign(dones_flag[i] + _d)
                state[i] = _v
                visual_state[i] = _vs
                if policy_mode == 'off-policy':
                    if train_step[i] % off_policy_train_interval == 0:
                        models[i].learn(episode=episode, train_step=train_step)
                    train_step[i] += 1
                    if train_step[i] % save_frequency == 0:
                        models[i].save_checkpoint(train_step=train_step[i], episode=episode, frame_step=frame_step)

            frame_step += min_of_all_agents
            if 0 < max_train_step < min(train_step) or 0 < max_frame_step < frame_step:
                for i in range(env.brain_num):
                    models[i].save_checkpoint(train_step=train_step[i], episode=episode, frame_step=frame_step)
                logger.info(f'End Training, learn step: {train_step}, frame_step: {frame_step}')
                return

            if all([all(dones_flag[i]) for i in range(env.brain_num)]):
                if last_done_step == -1:
                    last_done_step = step
                if policy_mode == 'off-policy':
                    break

            if step >= max_step_per_episode:
                break

        for i in range(env.brain_num):
            sma[i].update(rewards[i])
            if policy_mode == 'on-policy':
                models[i].learn(episode=episode, train_step=train_step)
                train_step[i] += 1
                if train_step[i] % save_frequency == 0:
                    models[i].save_checkpoint(train_step=train_step[i], episode=episode, frame_step=frame_step)
            models[i].writer_summary(
                episode,
                reward_mean=rewards[i].mean(),
                reward_min=rewards[i].min(),
                reward_max=rewards[i].max(),
                step=last_done_step,
                **sma[i].rs
            )
        print_func('-' * 40, out_time=True)
        print_func(f'episode {episode:3d} | step {step:4d} | last_done_step {last_done_step:4d}')
        for i, bn in enumerate(env.brain_names):
            print_func(f'{bn} reward: {arrprint(rewards[i], 2)}')

        if add_noise2buffer and episode % add_noise2buffer_episode_interval == 0:
            unity_no_op(env, models, print_func=print_func, pre_fill_steps=add_noise2buffer_steps, prefill_choose=False, real_done=real_done,
                        desc='adding noise')


def unity_no_op(env, models, print_func, pre_fill_steps, prefill_choose, real_done, desc='Pre-filling'):
    '''
    Interact with the environment but do not perform actions. Prepopulate the ReplayBuffer.
    Make sure steps is greater than n-step if using any n-step ReplayBuffer.
    '''
    assert isinstance(pre_fill_steps, int) and pre_fill_steps >= 0, 'no_op.steps must have type of int and larger than/equal 0'
    state, visual_state, action = zeros_initializer(env.brain_num, 3)

    [model.reset() for model in models]
    ObsRewDone = zip(*env.reset())
    for i, (_v, _vs, _r, _d, _info) in enumerate(ObsRewDone):
        state[i] = _v
        visual_state[i] = _vs

    for _ in trange(0, pre_fill_steps, min(env.brain_agents) + 1, unit_scale=min(env.brain_agents) + 1, ncols=80, desc=desc, bar_format=bar_format):
        if prefill_choose:
            for i in range(env.brain_num):
                action[i] = models[i].choose_action(s=state[i], visual_s=visual_state[i])
        else:
            action = env.random_action()
        actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.brain_names)}
        ObsRewDone = zip(*env.step(actions))
        for i, (_v, _vs, _r, _d, _info) in enumerate(ObsRewDone):
            models[i].no_op_store(
                s=state[i],
                visual_s=visual_state[i],
                a=action[i],
                r=_r,
                s_=_v,
                visual_s_=_vs,
                done=_info['real_done'] if real_done else _d
            )
            models[i].partial_reset(_d)
            state[i] = _v
            visual_state[i] = _vs


def unity_inference(env, models, episodes):
    """
    inference mode. algorithm model will not be train, only used to show agents' behavior
    """
    action = zeros_initializer(env.brain_num, 1)

    for episode in range(episodes):
        [model.reset() for model in models]
        ObsRewDone = zip(*env.reset())
        while True:
            for i, (_v, _vs, _r, _d, _info) in enumerate(ObsRewDone):
                action[i] = models[i].choose_action(s=_v, visual_s=_vs, evaluation=True)
                models[i].partial_reset(_d)
            actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.brain_names)}
            ObsRewDone = zip(*env.step(actions))


def ma_unity_no_op(env, model, print_func, pre_fill_steps, prefill_choose, desc='Pre-filling', real_done=True):
    assert isinstance(pre_fill_steps, int), 'multi-agent no_op.steps must have type of int'

    data_change_func = multi_agents_data_preprocess(env.env_copys, env.brain_controls)
    action_reshape_func = multi_agents_action_reshape(env.env_copys, env.brain_controls)
    model.reset()

    # [s(s_brain1(agent1, agent2, ...), s_brain2, ...), visual_s, r, done, info]
    s, visual_s, _, _, _ = env.reset()
    # [total_agents, batch, dimension]
    s, visual_s = map(data_change_func, [s, visual_s])

    for _ in trange(0, pre_fill_steps, env.env_copys, unit_scale=env.env_copys, ncols=80, desc=desc, bar_format=bar_format):
        if prefill_choose:
            action = model.choose_action(s=s, visual_s=visual_s)    # [total_agents, batch, dimension]
            action = action_reshape_func(action)
        else:
            action = env.random_action()
        actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.brain_names)}
        s_, visual_s_, r, done, info = env.step(actions)
        if real_done:
            done = [b['real_done'] for b in info]

        action, r, done, s_, visual_s_ = map(data_change_func, [action, r, done, s_, visual_s_])
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
        s = s_
        visual_s = visual_s_


def ma_unity_train(env, model, print_func,
                   begin_train_step, begin_frame_step, begin_episode, max_train_step, max_frame_step,
                   off_policy_train_interval, moving_average_episode,
                   save_frequency, max_step_per_episode, max_train_episode, policy_mode, real_done=True):
    assert policy_mode == 'off-policy', "multi-agents algorithms now support off-policy only."

    frame_step = begin_frame_step
    train_step = begin_train_step

    data_change_func = multi_agents_data_preprocess(env.env_copys, env.brain_controls)
    action_reshape_func = multi_agents_action_reshape(env.env_copys, env.brain_controls)
    agents_num_per_copy = sum(env.brain_controls)

    sma = [SMA(moving_average_episode) for _ in range(agents_num_per_copy)]

    for episode in range(begin_episode, max_train_episode):

        dones_flag = np.zeros(env.env_copys)
        rewards = np.zeros((agents_num_per_copy, env.env_copys))

        model.reset()
        s, visual_s, _, _, _ = env.reset()
        s, visual_s = map(data_change_func, [s, visual_s])

        step = 0
        last_done_step = -1
        while True:
            action = model.choose_action(s=s, visual_s=visual_s)    # [total_agents, batch, dimension]
            action = action_reshape_func(action)
            actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.brain_names)}
            s_, visual_s_, r, done, info = env.step(actions)    # [Brains, Agents, Dims]
            step += 1

            if real_done:
                done = [b['real_done'] for b in info]

            # [Agents_perCopy, Copys, Dims]
            action, r, done, s_, visual_s_ = map(data_change_func, [action, r, done, s_, visual_s_])
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
            s = s_
            visual_s = visual_s_

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

        print_func('-' * 40, out_time=True)
        print_func(f'episode {episode:3d} | step {step:4d} | last_done_step {last_done_step:4d}')
        for i in range(agents_num_per_copy):
            print_func(f'agent {i} reward: {arrprint(rewards[i], 2)}')


def ma_unity_inference(env, model, episodes):
    """
    inference mode. algorithm model will not be train, only used to show agents' behavior
    """
    data_change_func = multi_agents_data_preprocess(env.env_copys, env.brain_controls)
    action_reshape_func = multi_agents_action_reshape(env.env_copys, env.brain_controls)
    for episode in range(episodes):
        model.reset()
        s, visual_s, _, _, _ = env.reset()
        while True:
            action = model.choose_action(s=s, visual_s=visual_s, evaluation=True)    # [total_agents, batch, dimension]
            action = action_reshape_func(action)
            actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.brain_names)}
            s, visual_s, _, _, _ = env.step(actions)
