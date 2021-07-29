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


def train(env, model,
          print_func: Callable[[str], None],
          begin_train_step: int,
          begin_frame_step: int,
          begin_episode: int,
          save_frequency: int,
          episode_length: int,
          max_train_episode: int,
          policy_mode: str,
          moving_average_episode: int,
          off_policy_train_interval: int,
          max_train_step: int,
          max_frame_step: int,
          render: bool,
          render_episode: int,
          reset_config: dict,
          step_config: dict,

          off_policy_step_eval_episodes: int,
          off_policy_eval_interval: int) -> NoReturn:
    """
    TODO: Annotation
    Train loop. Execute until episode reaches its maximum or press 'ctrl+c' artificially.
    Inputs:
        env:                    Environment for interaction.
        model:                  all model for this training task.
        save_frequency:         how often to save checkpoints.
        episode_length:         maximum number of steps for an episode.
    """
    frame_step = begin_frame_step
    train_step = begin_train_step

    recoder = SimpleMovingAverageRecoder(n_copys=env.n_copys,
                                         n_agents=env.n_agents,
                                         gamma=0.99,
                                         verbose=True,
                                         length=moving_average_episode)

    for episode in range(begin_episode, max_train_episode):
        model.reset()
        obs = env.reset(reset_config={})
        recoder.episode_reset(episode=episode)
        for _ in range(episode_length):
            if render or episode > render_episode:
                env.render(record=False)
            actions = model(obs=obs)
            rets = env.step(actions, step_config={})
            expss = [BatchExperiences(obs=obs[i],
                                      action=actions[i],
                                      reward=ret.reward[:, np.newaxis],  # [B, ] => [B, 1]
                                      obs_=ret.obs,
                                      done=ret.done[:, np.newaxis])
                     for i, ret in enumerate(rets)]
            model.store_data(expss)
            model.partial_reset([ret.done for ret in rets])
            recoder.step_update(rewards=[ret.reward for ret in rets],
                                dones=[ret.done for ret in rets])
            obs = [ret.corrected_obs for ret in rets]

            if policy_mode == 'off-policy':
                if recoder.total_step % off_policy_train_interval == 0:
                    model.learn(episode=episode, train_step=train_step)
                    train_step += 1
                    if train_step % save_frequency == 0:
                        model.save(train_step=train_step, episode=episode, frame_step=frame_step)
                # if off_policy_eval_interval > 0 and train_step % off_policy_eval_interval == 0:
                #     evaluate(deepcopy(env), model, reset_config, step_config, train_step, off_policy_step_eval_episodes, episode_length)

            frame_step += env.n_copys
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
        model.write_summaries(episode, recoder.summary_dict(title='Train'))
        print_func(str(recoder), out_time=True)


def prefill(env, model,
            pre_fill_steps: int,
            prefill_choose: bool,
            reset_config: dict,
            step_config: dict,
            desc: str = 'Pre-filling') -> NoReturn:

    assert isinstance(pre_fill_steps, int) and pre_fill_steps >= 0, 'prefill.steps must have type of int and larger than/equal 0'

    if pre_fill_steps == 0:
        return
    model.reset()
    obs = env.reset(reset_config={})

    for _ in trange(0, pre_fill_steps, env.n_copys, unit_scale=env.n_copys, ncols=80, desc=desc, bar_format=bar_format):

        if prefill_choose:
            actions = model(obs=obs, evaluation=True)
        else:
            actions = env.random_action()
        rets = env.step(actions, step_config={})
        expss = [BatchExperiences(obs=obs[i],
                                  action=actions[i],
                                  reward=ret.reward[:, np.newaxis],  # [B, ] => [B, 1]
                                  obs_=ret.obs,
                                  done=ret.done[:, np.newaxis])
                 for i, ret in enumerate(rets)]
        model.prefill_store(expss)
        model.partial_reset([ret.done for ret in rets])
        obs = [ret.corrected_obs for ret in rets]


def inference(env, model,
              print_func: Callable[[str], None],
              reset_config: dict,
              step_config: dict,
              episodes: int) -> NoReturn:
    """
    inference mode. algorithm model will not be train, only used to show agents' behavior
    """

    recoder = SimpleMovingAverageRecoder(n_copys=env.n_copys,
                                         n_agents=env.n_agents,
                                         gamma=0.99,
                                         verbose=True,
                                         length=moving_average_episode)
    for episode in range(episodes):
        model.reset()
        obs = env.reset(reset_config={})
        recoder.episode_reset(episode=episode)

        while True:
            env.render(record=False)
            actions = model(obs=obs, evaluation=True)
            rets = env.step(actions, step_config={})
            model.partial_reset([ret.done for ret in rets])
            obs = [ret.corrected_obs for ret in rets]
            recoder.step_update(rewards=[ret.reward for ret in rets],
                                dones=[ret.done for ret in rets])
            if recoder.is_all_done:
                break

        recoder.episode_end()
        print_func(str(recoder), out_time=True)


def evaluate(env, model,
             reset_config: dict,
             step_config: dict,
             train_step: int,
             episodes_num: int,
             episode_length: int) -> NoReturn:
    '''
    TODO: fix bugs
    '''
    cell_state = model.get_cell_state()  # 暂存训练时候的RNN隐状态
    recoder = SimpleMovingAverageRecoder(n_copys=env.n_copys,
                                         n_agents=env.n_agents,
                                         gamma=0.99,
                                         verbose=True,
                                         length=episodes_num)

    for _ in trange(episodes_num, ncols=80, desc='evaluating', bar_format=bar_format):
        model.reset()
        obs = env.reset(reset_config={})
        recoder.episode_reset(episode=episode)

        for _ in range(episode_length):
            actions = model(obs=obs, evaluation=True)
            rets = env.step(actions, step_config={})
            model.partial_reset([ret.done for ret in rets])
            obs = [ret.corrected_obs for ret in rets]
            recoder.step_update(rewards=[ret.reward for ret in rets],
                                dones=[ret.done for ret in rets])
            if recoder.is_all_done:
                break
        recoder.episode_end()
    model.write_summaries(train_step, recoder.summary_dict(title='Eval'))
    model.set_cell_state(cell_state)
