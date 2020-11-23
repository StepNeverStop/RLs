#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from tqdm import trange
from typing import (Callable,
                    NoReturn)

from rls.utils.np_utils import (SMA,
                                arrprint)
from rls.utils.list_utils import zeros_initializer
from rls.utils.mlagents_utils import (multi_agents_data_preprocess,
                                      multi_agents_action_reshape)
from rls.utils.logging_utils import get_logger

logger = get_logger(__name__)
bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'


def unity_train(env, models,
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
        models:                 all models for this training task.
        save_frequency:         how often to save checkpoints.
        max_step_per_episode:   maximum number of steps for an episode.
        resampling_interval:    how often to resample parameters for env reset.
    Variables:
        group_names:    a list of group names set in Unity.
        state: store    a list of states for each group. each item contain a list of states for each agents that controlled by the same group.
        visual_state:   store a list of visual state information for each group.
        action:         store a list of actions for each group.
        dones_flag:     store a list of 'done' for each group. use for judge whether an episode is finished for every agents.
        rewards:        use to record rewards of agents for each group.
    """

    state, visual_state, action, dones_flag, rewards = zeros_initializer(env.group_num, 5)
    sma = [SMA(moving_average_episode) for i in range(env.group_num)]
    frame_step = begin_frame_step
    min_of_all_agents = min(env.group_agents)
    train_step = [begin_train_step for _ in range(env.group_num)]

    for episode in range(begin_episode, max_train_episode):
        [model.reset() for model in models]
        ObsRewDone = zip(*env.reset())
        for i, (_v, _vs, _r, _d, _info, _corrected_v, _correcred_vs) in enumerate(ObsRewDone):
            dones_flag[i] = np.zeros(env.group_agents[i])
            rewards[i] = np.zeros(env.group_agents[i])
            state[i] = _corrected_v
            visual_state[i] = _correcred_vs
        step = 0
        last_done_step = -1
        while True:
            step += 1
            for i in range(env.group_num):
                action[i] = models[i].choose_action(s=state[i], visual_s=visual_state[i])
            actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.group_names)}
            ObsRewDone = zip(*env.step(actions))

            for i, (_v, _vs, _r, _d, _info, _corrected_v, _correcred_vs) in enumerate(ObsRewDone):
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
                state[i] = _corrected_v
                visual_state[i] = _correcred_vs
                if policy_mode == 'off-policy':
                    if train_step[i] % off_policy_train_interval == 0:
                        models[i].learn(episode=episode, train_step=train_step[i])
                    train_step[i] += 1
                    if train_step[i] % save_frequency == 0:
                        models[i].save_checkpoint(train_step=train_step[i], episode=episode, frame_step=frame_step)

            frame_step += min_of_all_agents
            if 0 < max_train_step < min(train_step) or 0 < max_frame_step < frame_step:
                for i in range(env.group_num):
                    models[i].save_checkpoint(train_step=train_step[i], episode=episode, frame_step=frame_step)
                logger.info(f'End Training, learn step: {train_step}, frame_step: {frame_step}')
                return

            if all([all(dones_flag[i]) for i in range(env.group_num)]):
                if last_done_step == -1:
                    last_done_step = step
                if policy_mode == 'off-policy':
                    break

            if step >= max_step_per_episode:
                break

        for i in range(env.group_num):
            sma[i].update(rewards[i])
            if policy_mode == 'on-policy':
                models[i].learn(episode=episode, train_step=train_step[i])
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
        print_func(f'Eps {episode:3d} | S {step:4d} | LDS {last_done_step:4d}', out_time=True)
        for i, gn in enumerate(env.group_names):
            print_func(f'{gn} R: {arrprint(rewards[i], 2)}')

        if add_noise2buffer and episode % add_noise2buffer_episode_interval == 0:
            unity_no_op(env, models, pre_fill_steps=add_noise2buffer_steps, prefill_choose=False, real_done=real_done,
                        desc='adding noise')


def unity_no_op(env, models,
                pre_fill_steps: int,
                prefill_choose: bool,
                real_done: bool,
                desc: str = 'Pre-filling') -> NoReturn:
    '''
    Interact with the environment but do not perform actions. Prepopulate the ReplayBuffer.
    Make sure steps is greater than n-step if using any n-step ReplayBuffer.
    '''
    assert isinstance(pre_fill_steps, int) and pre_fill_steps >= 0, 'no_op.steps must have type of int and larger than/equal 0'
    min_of_all_agents = min(env.group_agents)

    if pre_fill_steps == 0:
        return

    state, visual_state, action = zeros_initializer(env.group_num, 3)

    [model.reset() for model in models]
    ObsRewDone = zip(*env.reset())
    for i, (_v, _vs, _r, _d, _info, _corrected_v, _correcred_vs) in enumerate(ObsRewDone):
        state[i] = _corrected_v
        visual_state[i] = _correcred_vs

    for _ in trange(0, pre_fill_steps, min_of_all_agents, unit_scale=min_of_all_agents, ncols=80, desc=desc, bar_format=bar_format):
        if prefill_choose:
            for i in range(env.group_num):
                action[i] = models[i].choose_action(s=state[i], visual_s=visual_state[i])
        else:
            action = env.random_action()
        actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.group_names)}
        ObsRewDone = zip(*env.step(actions))
        for i, (_v, _vs, _r, _d, _info, _corrected_v, _correcred_vs) in enumerate(ObsRewDone):
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
            state[i] = _corrected_v
            visual_state[i] = _correcred_vs


def unity_inference(env, models,
                    episodes: int) -> NoReturn:
    """
    inference mode. algorithm model will not be train, only used to show agents' behavior
    """
    action = zeros_initializer(env.group_num, 1)

    for episode in range(episodes):
        [model.reset() for model in models]
        ObsRewDone = zip(*env.reset())
        while True:
            for i, (_v, _vs, _r, _d, _info, _corrected_v, _correcred_vs) in enumerate(ObsRewDone):
                action[i] = models[i].choose_action(s=_corrected_v, visual_s=_correcred_vs, evaluation=True)
                models[i].partial_reset(_d)
            actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.group_names)}
            ObsRewDone = zip(*env.step(actions))


def ma_unity_no_op(env, model,
                   pre_fill_steps: int,
                   prefill_choose: bool,
                   desc: str = 'Pre-filling',
                   real_done: bool = True) -> NoReturn:
    assert isinstance(pre_fill_steps, int) and pre_fill_steps >= 0, 'multi-agent no_op.steps must have type of int'

    if pre_fill_steps == 0:
        return

    data_change_func = multi_agents_data_preprocess(env.env_copys, env.group_controls)
    action_reshape_func = multi_agents_action_reshape(env.env_copys, env.group_controls)
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
        actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.group_names)}
        s_, visual_s_, r, done, info = env.step(actions)
        if real_done:
            done = [g['real_done'] for g in info]

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

    data_change_func = multi_agents_data_preprocess(env.env_copys, env.group_controls)
    action_reshape_func = multi_agents_action_reshape(env.env_copys, env.group_controls)
    agents_num_per_copy = sum(env.group_controls)

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
            actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.group_names)}
            s_, visual_s_, r, done, info = env.step(actions)    # [Brains, Agents, Dims]
            step += 1

            if real_done:
                done = [g['real_done'] for g in info]

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

        print_func(f'Eps {episode:3d} | S {step:4d} | LDS {last_done_step:4d}', out_time=True)
        for i in range(agents_num_per_copy):
            print_func(f'Agent {i} R: {arrprint(rewards[i], 2)}')


def ma_unity_inference(env, model,
                       episodes: int) -> NoReturn:
    """
    inference mode. algorithm model will not be train, only used to show agents' behavior
    """
    data_change_func = multi_agents_data_preprocess(env.env_copys, env.group_controls)
    action_reshape_func = multi_agents_action_reshape(env.env_copys, env.group_controls)
    for episode in range(episodes):
        model.reset()
        s, visual_s, _, _, _ = env.reset()
        while True:
            action = model.choose_action(s=s, visual_s=visual_s, evaluation=True)    # [total_agents, batch, dimension]
            action = action_reshape_func(action)
            actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.group_names)}
            s, visual_s, _, _, _ = env.step(actions)
