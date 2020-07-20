import logging
import numpy as np

from tqdm import trange
from copy import deepcopy
from utils.np_utils import SMA, arrprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("common.train.gym")
bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'


def init_variables(env):
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
            [np.array([[]] * env.n, dtype=np.float32), np.array([[]] * env.n, dtype=np.float32)],
            [np.array([[]] * env.n, dtype=np.float32), np.array([[]] * env.n, dtype=np.float32)])


def gym_train(env, model, print_func,
              begin_train_step, begin_frame_step, begin_episode, render, render_episode,
              save_frequency, max_step_per_episode, max_train_episode, eval_while_train, max_eval_episode,
              off_policy_step_eval_episodes, off_policy_train_interval,
              policy_mode, moving_average_episode, add_noise2buffer, add_noise2buffer_episode_interval, add_noise2buffer_steps,
              off_policy_eval_interval, max_train_step, max_frame_step):
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
        dones_flag = np.full(env.n, False)
        step = 0
        r = np.zeros(env.n)
        last_done_step = -1
        while True:
            step += 1
            if render or episode > render_episode:
                env.render(record=False)
            action = model.choose_action(s=state[0], visual_s=state[1])
            new_state[i], reward, done, info, correct_new_state = env.step(action)
            unfinished_index = np.where(dones_flag == False)[0]
            dones_flag += done
            r[unfinished_index] += reward[unfinished_index]
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
                    gym_step_eval(deepcopy(env), train_step, model, off_policy_step_eval_episodes, max_step_per_episode)

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

        sma.update(r)
        if policy_mode == 'on-policy':
            model.learn(episode=episode, train_step=train_step)
            train_step += 1
            if train_step % save_frequency == 0:
                model.save_checkpoint(train_step=train_step, episode=episode, frame_step=frame_step)
        model.writer_summary(
            episode,
            reward_mean=r.mean(),
            reward_min=r.min(),
            reward_max=r.max(),
            step=last_done_step,
            **sma.rs
        )
        print_func('-' * 40, out_time=True)
        print_func(f'Episode: {episode:3d} | step: {step:4d} | last_done_step {last_done_step:4d} | rewards: {arrprint(r, 2)}')

        if add_noise2buffer and episode % add_noise2buffer_episode_interval == 0:
            gym_no_op(env, model, pre_fill_steps=add_noise2buffer_steps, print_func=print_func, prefill_choose=False, desc='adding noise')

        if eval_while_train and env.reward_threshold is not None:
            if r.max() >= env.reward_threshold:
                print_func(f'-------------------------------------------Evaluate episode: {episode:3d}--------------------------------------------------')
                gym_evaluate(env, model, max_step_per_episode, max_eval_episode, print_func)


def gym_step_eval(env, step, model, episodes_num, max_step_per_episode):
    '''
    1个环境的推断模式
    '''
    cs = model.get_cell_state()  # 暂存训练时候的RNN隐状态

    i, state, _ = init_variables(env)
    ret = 0.
    ave_steps = 0.
    for _ in trange(episodes_num, ncols=80, desc='evaluating', bar_format=bar_format):
        model.reset()
        state[i] = env.reset()
        r = 0.
        step = 0
        while True:
            action = model.choose_action(s=state[0], visual_s=state[1], evaluation=True)
            _, reward, done, info, state[i] = env.step(action)
            model.partial_reset(done)
            reward = reward[0]
            done = done[0]
            r += reward
            step += 1
            if done or step > max_step_per_episode:
                ret += r
                ave_steps += step
                break

    model.writer_summary(
        step,
        eval_return=ret / episodes_num,
        eval_ave_step=ave_steps // episodes_num,
    )
    model.set_cell_state(cs)
    del env


def gym_evaluate(env, model, max_step_per_episode, max_eval_episode, print_func):
    i, state, _ = init_variables(env)
    total_r = np.zeros(env.n)
    total_steps = np.zeros(env.n)

    for _ in trange(max_eval_episode, ncols=80, desc='evaluating', bar_format=bar_format):
        model.reset()
        state[i] = env.reset()
        dones_flag = np.full(env.n, False)
        steps = np.zeros(env.n)
        r = np.zeros(env.n)
        while True:
            action = model.choose_action(s=state[0], visual_s=state[1], evaluation=True)  # In the future, this method can be combined with choose_action
            _, reward, done, info, state[i] = env.step(action)
            model.partial_reset(done)
            unfinished_index = np.where(dones_flag == False)
            dones_flag += done
            r[unfinished_index] += reward[unfinished_index]
            steps[unfinished_index] += 1
            if all(dones_flag) or any(steps >= max_step_per_episode):
                break
        total_r += r
        total_steps += steps
    average_r = total_r.mean() / max_eval_episode
    average_step = int(total_steps.mean() / max_eval_episode)
    solved = True if average_r >= env.reward_threshold else False
    print_func(f'evaluate number: {max_eval_episode:3d} | average step: {average_step} | average reward: {average_r} | SOLVED: {solved}')
    print_func('----------------------------------------------------------------------------------------------------------------------------')


def gym_no_op(env, model, print_func, pre_fill_steps, prefill_choose, desc='Pre-filling'):
    assert isinstance(pre_fill_steps, int) and pre_fill_steps >= 0, 'no_op.steps must have type of int and larger than/equal 0'

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


def gym_inference(env, model, episodes):
    i, state, _ = init_variables(env)
    for episode in range(episodes):
        step = 0
        model.reset()
        state[i] = env.reset()
        dones = np.full(shape=(env.n,), fill_value=False, dtype=np.bool)
        returns = np.zeros(shape=(env.n), dtype=np.float32)
        while True:
            env.render(record=False)
            action = model.choose_action(s=state[0], visual_s=state[1], evaluation=True)
            step += 1
            _, reward, done, info, state[i] = env.step(action)
            unfinished_index = np.where(dones == False)[0]
            returns[unfinished_index] += reward[unfinished_index]
            dones += done
            model.partial_reset(done)
            if dones.all():
                logger.info(f'episode: {episode:4d}, returns: min {returns.min():6.2f}, mean {returns.mean():6.2f}, max {returns.max():6.2f}')
                break

            if step % 1000 == 0:
                logger.info(f'step: {step}')
