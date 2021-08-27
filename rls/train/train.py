#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from tqdm import trange
from copy import deepcopy
from typing import (Tuple,
                    List,
                    Callable,
                    NoReturn)

from rls.common.recorder import SimpleMovingAverageRecoder
from rls.common.specs import Data
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)
bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'


def train(env, agent,
          print_func: Callable[[str], None],
          episode_length: int,
          moving_average_episode: int,
          render: bool,
          reset_config: dict,
          step_config: dict) -> NoReturn:
    """
    TODO: Annotation
    Train loop. Execute until episode reaches its maximum or press 'ctrl+c' artificially.
    Inputs:
        env:                    Environment for interaction.
        agent:                  all agent for this training task.
        episode_length:         maximum number of steps for an episode.
    """
    recorder = SimpleMovingAverageRecoder(n_copys=env.n_copys,
                                          agent_ids=env.agent_ids,
                                          gamma=0.99,
                                          verbose=True,
                                          length=moving_average_episode)

    agent.setup(is_train_mode=True, store=True)

    while agent.still_learn:
        agent.episode_reset()
        recorder.episode_reset()
        obs = env.reset(reset_config={})

        for _ in range(episode_length):
            if render:
                env.render(record=False)
            acts = agent(obs=obs)
            env_rets = env.step(
                {id: acts[id].action for id in env.agent_ids}, step_config={})
            agent.episode_step(obs, acts, env_rets)
            recorder.episode_step(rewards={id: env_rets[id].reward for id in env.agent_ids},
                                  dones={id: env_rets[id].done for id in env.agent_ids})
            obs = {id: env_rets[id].obs for id in env.agent_ids}
            obs['global'] = env_rets['global']
            if recorder.is_all_done and agent.policy_mode == 'off-policy':
                break

        recorder.episode_end()
        agent.episode_end()
        agent.write_recorder_summaries(recorder.summary_dict(title='Train'))
        print_func(str(recorder), out_time=True)

    # TODO: print training end info.


def prefill(env, agent,
            prefill_steps: int,
            reset_config: dict,
            step_config: dict,
            desc: str = 'Pre-filling') -> NoReturn:

    assert isinstance(prefill_steps, int) \
        and prefill_steps >= 0, 'prefill.steps must have type of int and larger than/equal 0'

    if agent.policy_mode == 'on-policy':
        return

    agent.setup(is_train_mode=False, store=True)

    agent.episode_reset()
    obs = env.reset(reset_config={})

    for _ in trange(0, prefill_steps, env.n_copys, unit_scale=env.n_copys, ncols=80, desc=desc, bar_format=bar_format):
        acts = agent.random_action()
        env_rets = env.step(
            {id: acts[id].action for id in env.agent_ids}, step_config={})
        agent.episode_step(obs, acts, env_rets)
        obs = {id: env_rets[id].obs for id in env.agent_ids}
        obs['global'] = env_rets['global']


def inference(env, agent,
              print_func: Callable[[str], None],
              moving_average_episode: int,
              reset_config: dict,
              step_config: dict,
              episodes: int) -> NoReturn:
    """
    inference mode. algorithm agent will not be train, only used to show agents' behavior
    """

    recorder = SimpleMovingAverageRecoder(n_copys=env.n_copys,
                                          agent_ids=env.agent_ids,
                                          gamma=0.99,
                                          verbose=True,
                                          length=moving_average_episode)
    agent.setup(is_train_mode=False, store=False)

    for episode in range(episodes):
        agent.episode_reset()
        obs = env.reset(reset_config={})
        recorder.episode_reset()

        while True:
            env.render(record=False)
            acts = agent(obs=obs)
            env_rets = env.step(
                {id: acts[id].action for id in env.agent_ids}, step_config={})
            agent.episode_step(obs, acts, env_rets)
            recorder.episode_step(rewards={id: env_rets[id].reward for id in env.agent_ids},
                                  dones={id: env_rets[id].done for id in env.agent_ids})
            obs = {id: env_rets[id].obs for id in env.agent_ids}
            obs['global'] = env_rets['global']
            if recorder.is_all_done:
                break
        recorder.episode_end()
        agent.episode_end()
        print_func(str(recorder), out_time=True)


def evaluate(env, agent,
             reset_config: dict,
             step_config: dict,
             episodes_num: int,
             episode_length: int) -> NoReturn:
    '''
    TODO: fix bugs
    '''
    recorder = SimpleMovingAverageRecoder(n_copys=env.n_copys,
                                          agent_ids=env.agent_ids,
                                          gamma=0.99,
                                          verbose=True,
                                          length=episodes_num)
    agent.setup(is_train_mode=False, store=False)

    for _ in trange(episodes_num, ncols=80, desc='evaluating', bar_format=bar_format):
        agent.episode_reset()
        obs = env.reset(reset_config={})
        recorder.episode_reset()

        for _ in range(episode_length):
            acts = agent(obs=obs)
            env_rets = env.step(
                {id: acts[id].action for id in env.agent_ids}, step_config={})
            agent.episode_step({id: env_rets[id].done for id in env.agent_ids})
            obs = {id: env_rets[id].obs for id in env.agent_ids}
            obs['global'] = env_rets['global']
            recorder.episode_step(rewards={id: env_rets[id].reward for id in env.agent_ids},
                                  dones={id: env_rets[id].done for id in env.agent_ids})
            if recorder.is_all_done:
                break
        recorder.episode_end()
        agent.episode_end()
        agent.write_recorder_summaries(recorder.summary_dict(title='Eval'))
