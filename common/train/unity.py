

import numpy as np

from tqdm import trange
from utils.np_utils import SMA, arrprint
from utils.list_utils import zeros_initializer

def unity_train(env, models, print_func,
                begin_episode, save_frequency, max_step, max_episode, policy_mode,
                moving_average_episode, add_noise2buffer, add_noise2buffer_episode_interval, add_noise2buffer_steps,
                total_step_control, max_total_step, real_done):
    """
    TODO: Annotation
    Train loop. Execute until episode reaches its maximum or press 'ctrl+c' artificially.
    Inputs:
        env:                    Environment for interaction.
        models:                 all models for this trianing task.
        save_frequency:         how often to save checkpoints.
        reset_config:           configuration to reset for Unity environment.
        max_step:               maximum number of steps for an episode.
        sampler_manager:        sampler configuration parameters for 'reset_config'.
        resampling_interval:    how often to resample parameters for env reset.
    Variables:
        brain_names:    a list of brain names set in Unity.
        state: store    a list of states for each brain. each item contain a list of states for each agents that controlled by the same brain.
        visual_state:   store a list of visual state information for each brain.
        action:         store a list of actions for each brain.
        dones_flag:     store a list of 'done' for each brain. use for judge whether an episode is finished for every agents.
        rewards:        use to record rewards of agents for each brain.
    """

    if total_step_control:
        max_episode = max_total_step

    state, visual_state, action, dones_flag, rewards = zeros_initializer(env.brain_num, 5)
    sma = [SMA(moving_average_episode) for i in range(env.brain_num)]
    total_step = 0

    for episode in range(begin_episode, max_episode):
        [model.reset() for model in models]
        ObsRewDone = env.reset()
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
            ObsRewDone = env.step(actions)

            for i, (_v, _vs, _r, _d, _info) in enumerate(ObsRewDone):
                unfinished_index = np.where(dones_flag[i] == False)[0]
                dones_flag[i] += _d
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
                rewards[i][unfinished_index] += _r[unfinished_index]
                state[i] = _v
                visual_state[i] = _vs
                if policy_mode == 'off-policy':
                    models[i].learn(episode=episode, step=1)

            total_step += 1
            if total_step_control and total_step > max_total_step:
                for i in range(env.brain_num):
                    models[i].save_checkpoint(episode)
                return

            if all([all(dones_flag[i]) for i in range(env.brain_num)]):
                if last_done_step == -1:
                    last_done_step = step
                if policy_mode == 'off-policy':
                    break

            if step >= max_step:
                break

        for i in range(env.brain_num):
            sma[i].update(rewards[i])
            if policy_mode == 'on-policy':
                models[i].learn(episode=episode, step=step)
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
            print_func(f'{bn} reward: {arrprint(rewards[i], 3)}')
        if episode % save_frequency == 0:
            for i in range(env.brain_num):
                models[i].save_checkpoint(episode)

        if add_noise2buffer and episode % add_noise2buffer_episode_interval == 0:
            unity_random_sample(env, models, print_func, steps=add_noise2buffer_steps, real_done=real_done)

def unity_random_sample(env, models, print_func, steps, real_done):
    state, visual_state = zeros_initializer(env.brain_num, 2)

    ObsRewDone = env.reset()
    for i, (_v, _vs, _r, _d, _info) in enumerate(ObsRewDone):
        state[i] = _v
        visual_state[i] = _vs

    for _ in range(steps):
        action = env.random_action()
        actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.brain_names)}
        ObsRewDone = env.step(actions)
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
            state[i] = _v
            visual_state[i] = _vs
    print_func('Noise added complete.')

def unity_no_op(env, models, print_func, pre_fill_steps, prefill_choose, real_done):
    '''
    Interact with the environment but do not perform actions. Prepopulate the ReplayBuffer.
    Make sure steps is greater than n-step if using any n-step ReplayBuffer.
    '''
    assert isinstance(pre_fill_steps, int) and pre_fill_steps >= 0, 'no_op.steps must have type of int and larger than/equal 0'
    state, visual_state, action = zeros_initializer(env.brain_num, 3)

    [model.reset() for model in models]
    ObsRewDone = env.reset()
    for i, (_v, _vs, _r, _d, _info) in enumerate(ObsRewDone):
        state[i] = _v
        visual_state[i] = _vs

    tqdm_bar = trange(0, pre_fill_steps, min(env.brain_agents) + 1)
    for step in tqdm_bar:
        tqdm_bar.set_description('Pre-filling')

        if prefill_choose:
            for i in range(env.brain_num):
                action[i] = models[i].choose_action(s=state[i], visual_s=visual_state[i])
        else:
            action = env.random_action()
        actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.brain_names)}
        ObsRewDone = env.step(actions)
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

def unity_inference(env, models):
    """
    inference mode. algorithm model will not be train, only used to show agents' behavior
    """
    action = zeros_initializer(env.brain_num, 1)

    while True:
        [model.reset() for model in models]
        ObsRewDone = env.reset()
        while True:
            for i, (_v, _vs, _r, _d, _info) in enumerate(ObsRewDone):
                action[i] = models[i].choose_action(s=_v, visual_s=_vs, evaluation=True)
                models[i].partial_reset(_d)
            actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.brain_names)}
            ObsRewDone = env.step(actions)

def ma_unity_no_op(env, models, buffer, print_func, pre_fill_steps, prefill_choose):
    assert isinstance(pre_fill_steps, int), 'multi-agent no_op.steps must have type of int'

    if pre_fill_steps < buffer.batch_size:
        pre_fill_steps = buffer.batch_size
    state, action, reward, next_state, dones = zeros_initializer(env.brain_num, 5)
    ObsRewDone = env.reset()
    for i, (_v, _vs, _r, _d, _info) in enumerate(ObsRewDone):
        state[i] = _v

    for i in range(env.brain_num):
        # initialize actions to zeros
        if env.is_continuous[i]:
            action[i] = np.zeros((env.brain_agents[i], env.a_dim[i][0]), dtype=np.int32)
        else:
            action[i] = np.zeros((env.brain_agents[i], 1), dtype=np.int32)

    a = [np.asarray(e) for e in zip(*action)]
    for step in range(pre_fill_steps):
        print_func(f'no op step {step}')
        for i in range(env.brain_num):
            if prefill_choose:
                action[i] = models[i].choose_action(s=state[i])
        actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.brain_names)}
        ObsRewDone = env.step(actions)
        for i, (_v, _vs, _r, _d, _info) in enumerate(ObsRewDone):
            reward[i] = _r[:, np.newaxis]
            next_state[i] = _vs
            dones[i] = _d[:, np.newaxis]

        def func(x): return [np.asarray(e) for e in zip(*x)]
        s, a, r, s_, done = map(func, [state, action, reward, next_state, dones])
        buffer.add(s, a, r, s_, done)
        for i in range(env.brain_num):
            state[i] = next_state[i]

def ma_unity_train(env, models, buffer, print_func,
                   begin_episode, save_frequency, max_step, max_episode, policy_mode):
    assert policy_mode == 'off-policy', "multi-agents algorithms now support off-policy only."

    batch_size = buffer.batch_size
    state, action, new_action, next_action, reward, next_state, dones, dones_flag, rewards = zeros_initializer(env.brain_num, 9)

    for episode in range(begin_episode, max_episode):
        ObsRewDone = env.reset()
        for i, (_v, _vs, _r, _d, _info) in enumerate(ObsRewDone):
            dones_flag[i] = np.zeros(env.brain_agents[i])
            rewards[i] = np.zeros(env.brain_agents[i])
            state[i] = _v
        step = 0
        last_done_step = -1
        while True:
            step += 1
            for i in range(env.brain_num):
                action[i] = models[i].choose_action(s=state[i])
            actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.brain_names)}
            ObsRewDone = env.step(actions)

            for i, (_v, _vs, _r, _d, _info) in enumerate(ObsRewDone):
                reward[i] = _r[:, np.newaxis]
                next_state = _v
                dones[i] = _d[:, np.newaxis]
                unfinished_index = np.where(dones_flag[i] == False)[0]
                dones_flag[i] += _d
                rewards[i][unfinished_index] += _r[unfinished_index]

            def func(x): return [np.asarray(e) for e in zip(*x)]
            s, a, r, s_, done = map(func, [state, action, reward, next_state, dones])
            buffer.add(s, a, r, s_, done)

            for i in range(env.brain_num):
                state[i] = next_state[i]

            s, a, r, s_, done = buffer.sample()
            for i, _ in enumerate(env.brain_names):
                next_action[i] = models[i].get_target_action(s=s_[:, i])
                new_action[i] = models[i].choose_action(s=s[:, i], evaluation=True)
            a_ = np.asarray([np.asarray(e) for e in zip(*next_action)])
            if policy_mode == 'off-policy':
                for i in range(env.brain_num):
                    models[i].learn(
                        episode=episode,
                        ap=np.asarray([np.asarray(e) for e in zip(*next_action[:i])]).reshape(batch_size, -1) if i != 0 else np.zeros((batch_size, 0)),
                        al=np.asarray([np.asarray(e) for e in zip(*next_action[-(env.brain_num - i - 1):])]
                                        ).reshape(batch_size, -1) if env.brain_num - i != 1 else np.zeros((batch_size, 0)),
                        ss=s.reshape(batch_size, -1),
                        ss_=s_.reshape(batch_size, -1),
                        aa=a.reshape(batch_size, -1),
                        aa_=a_.reshape(batch_size, -1),
                        s=s[:, i],
                        r=r[:, i]
                    )

            if all([all(dones_flag[i]) for i in range(env.brain_num)]):
                if last_done_step == -1:
                    last_done_step = step
                if policy_mode == 'off-policy':
                    break

            if step >= max_step:
                break

        for i in range(env.brain_num):
            models[i].writer_summary(
                episode,
                total_reward=rewards[i].mean(),
                step=last_done_step
            )
        print_func('-' * 40, out_time=True)
        print_func(f'episode {episode:3d} | step {step:4d} last_done_step | {last_done_step:4d}')
        if episode % save_frequency == 0:
            for i in range(env.brain_num):
                models[i].save_checkpoint(episode)

def ma_unity_inference(env, models):
    """
    inference mode. algorithm model will not be train, only used to show agents' behavior
    """
    action = zeros_initializer(env.brain_num, 1)
    while True:
        ObsRewDone = env.reset()
        while True:
            for i, (_v, _vs, _r, _d, _info) in enumerate(ObsRewDone):
                action[i] = models[i].choose_action(s=_v, evaluation=True)
            actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(env.brain_names)}
            ObsRewDone = env.step(actions)