#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from tqdm import trange
from typing import (Callable,
                    NoReturn)

from rls.common.recoder import (SimpleMovingAverageRecoder,
                                SimpleMovingAverageMultiAgentRecoder)
from rls.common.specs import BatchExperiences
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)
bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'


def unity_train(env, model,
                print_func: Callable[[str], None],
                begin_train_step: int,
                begin_frame_step: int,
                begin_episode: int,
                save_frequency: int,
                episode_length: int,
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
        episode_length:   maximum number of steps for an episode.
        resampling_interval:    how often to resample parameters for env reset.
    """
    frame_step = begin_frame_step
    train_step = begin_train_step
    recoder = SimpleMovingAverageRecoder(n_copys=env._n_copys, gamma=0.99, verbose=True,
                                         length=moving_average_episode)

    for episode in range(begin_episode, max_train_episode):
        model.reset()
        ret = env.reset(reset_config={})
        recoder.episode_reset(episode=episode)

        for _ in range(episode_length):
            obs = ret.corrected_obs
            action = model(obs=obs)
            ret = env.step(action, step_config={})
            model.store_data(BatchExperiences(obs=obs,
                                              action=action,
                                              reward=ret.reward[:, np.newaxis],  # [B, ] => [B, 1]
                                              obs_=ret.obs,
                                              done=(ret.info['real_done'] if real_done else ret.done)[:, np.newaxis]))  # [B, ] => [B, 1]
            model.partial_reset(ret.done)
            recoder.step_update(rewards=ret.reward, dones=ret.done)

            if policy_mode == 'off-policy':
                if recoder.total_step % off_policy_train_interval == 0:
                    model.learn(episode=episode, train_step=train_step)
                    train_step += 1
                    if train_step % save_frequency == 0:
                        model.save(train_step=train_step, episode=episode, frame_step=frame_step)

            frame_step += env._n_copys
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
        model.write_summaries(episode, recoder.summary_dict)
        print_func(str(recoder), out_time=True)

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
    n = env._n_copys

    if pre_fill_steps == 0:
        return
    model.reset()
    ret = env.reset(reset_config={})

    for _ in trange(0, pre_fill_steps, n, unit_scale=n, ncols=80, desc=desc, bar_format=bar_format):
        obs = ret.corrected_obs
        if prefill_choose:
            action = model(obs=obs, evaluation=True)
        else:
            action = env.random_action()
        ret = env.step(action, step_config={})
        model.no_op_store(BatchExperiences(obs=obs,
                                           action=action,
                                           reward=ret.reward[:, np.newaxis],
                                           obs_=ret.obs,
                                           done=(ret.info['real_done'] if real_done else ret.done)[:, np.newaxis]))
        model.partial_reset(ret.done)


def unity_inference(env, model,
                    episodes: int) -> NoReturn:
    """
    inference mode. algorithm model will not be train, only used to show agents' behavior
    """

    for episode in range(episodes):
        model.reset()
        ret = env.reset(reset_config={})
        while True:
            action = model(obs=ret.corrected_obs,
                           evaluation=True)
            ret = env.step(action, step_config={})
            model.partial_reset(ret.done)


def ma_unity_no_op(env, model,
                   pre_fill_steps: int,
                   prefill_choose: bool,
                   real_done: bool,
                   desc: str = 'Pre-filling') -> NoReturn:
    assert isinstance(pre_fill_steps, int) and pre_fill_steps >= 0, 'no_op.steps must have type of int and larger than/equal 0'
    n = env._n_copys

    if pre_fill_steps == 0:
        return
    model.reset()
    rets = env.reset(is_single=False, reset_config={})

    for _ in trange(0, pre_fill_steps, n, unit_scale=n, ncols=80, desc=desc, bar_format=bar_format):
        pre_obss = [ret.corrected_obs for ret in rets]
        if prefill_choose:
            actions = model(obs=pre_obss, evaluation=True)
        else:
            actions = env.random_action(is_single=False)
        rets = env.step(actions, is_single=False, step_config={})
        expss = [
            BatchExperiences(obs=pre_obss[i],
                             action=actions[i],
                             reward=ret.reward[:, np.newaxis],
                             obs_=ret.obs,
                             done=(ret.info['real_done'] if real_done else ret.done)[:, np.newaxis])
            for i, ret in enumerate(rets)
        ]
        model.no_op_store(expss)
        model.partial_reset([ret.done for ret in rets])


def ma_unity_train(env, model,
                   print_func: Callable[[str], None],
                   begin_train_step: int,
                   begin_frame_step: int,
                   begin_episode: int,
                   save_frequency: int,
                   episode_length: int,
                   max_train_episode: int,
                   policy_mode: str,
                   moving_average_episode: int,
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
        episode_length:   maximum number of steps for an episode.
        resampling_interval:    how often to resample parameters for env reset.
    """
    frame_step = begin_frame_step
    train_step = begin_train_step
    recoder = SimpleMovingAverageMultiAgentRecoder(n_copys=env._n_copys,
                                                   n_agents=env.n_agents,
                                                   gamma=0.99,
                                                   verbose=True,
                                                   length=moving_average_episode)

    for episode in range(begin_episode, max_train_episode):
        model.reset()
        rets = env.reset(is_single=False, reset_config={})
        recoder.episode_reset(episode=episode)

        for _ in range(episode_length):
            pre_obss = [ret.corrected_obs for ret in rets]
            actions = model(obs=pre_obss)
            rets = env.step(actions, is_single=False, step_config={})

            expss = [
                BatchExperiences(obs=pre_obss[i],
                                 action=actions[i],
                                 reward=ret.reward[:, np.newaxis],
                                 obs_=ret.obs,
                                 done=(ret.info['real_done'] if real_done else ret.done)[:, np.newaxis])
                for i, ret in enumerate(rets)
            ]
            model.store_data(expss)
            model.partial_reset([ret.done for ret in rets])
            recoder.step_update(rewards=[ret.reward for ret in rets],
                                dones=[ret.done for ret in rets])

            if policy_mode == 'off-policy':
                if recoder.total_step % off_policy_train_interval == 0:
                    model.learn(episode=episode, train_step=train_step)
                    train_step += 1
                    if train_step % save_frequency == 0:
                        model.save(train_step=train_step, episode=episode, frame_step=frame_step)

            frame_step += env._n_copys
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
        model.write_summaries(episode, recoder.summary_dict)
        print_func(str(recoder), out_time=True)


def ma_unity_inference(env, model,
                       episodes: int) -> NoReturn:
    """
    inference mode. algorithm model will not be train, only used to show agents' behavior
    """

    for episode in range(episodes):
        model.reset()
        rets = env.reset(is_single=False, reset_config={})
        while True:
            actions = model(obs=[ret.corrected_obs for ret in rets],
                            evaluation=True)
            rets = env.step(actions, is_single=False, step_config={})
            model.partial_reset([ret.done for ret in rets])
