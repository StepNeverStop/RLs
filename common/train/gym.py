import logging
import numpy as np

from tqdm import trange
from utils.np_utils import SMA, arrprint

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("common.train.gym")


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
              begin_train_step, render, render_episode,
              save_frequency, max_step_per_episode, max_train_episode, eval_while_train, max_eval_episode,
              off_policy_step_eval, off_policy_step_eval_num,
              policy_mode, moving_average_episode, add_noise2buffer, add_noise2buffer_episode_interval, add_noise2buffer_steps,
              eval_interval, max_train_step, max_frame_step):
    """
    TODO: Annotation
    """

    i, state, new_state = init_variables(env)
    sma = SMA(moving_average_episode)
    frame_step = 0
    train_step = begin_train_step

    for episode in range(max_train_episode):
        model.reset()
        state[i] = env.reset()
        dones_flag = np.full(env.n, False)
        step = 0
        r = np.zeros(env.n)
        last_done_step = -1
        while True:
            step += 1
            r_tem = np.zeros(env.n)
            if render or episode > render_episode:
                env.render(record=False)
            action = model.choose_action(s=state[0], visual_s=state[1])
            new_state[i], reward, done, info, correct_new_state = env.step(action)
            unfinished_index = np.where(dones_flag == False)[0]
            dones_flag += done
            r_tem[unfinished_index] = reward[unfinished_index]
            r += r_tem
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
                model.learn(episode=episode, train_step=train_step, step=1)
                train_step += 1
                if train_step % save_frequency == 0:
                    model.save_checkpoint(train_step)
                if off_policy_step_eval and train_step % eval_interval == 0:
                    gym_step_eval(env.eval_env, train_step, model, off_policy_step_eval_num, max_step_per_episode)

            frame_step += env.n
            if 0 < max_train_step <= train_step or 0 < max_frame_step <= frame_step:
                model.save_checkpoint(train_step)
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
                model.save_checkpoint(train_step)
        model.writer_summary(
            episode,
            reward_mean=r.mean(),
            reward_min=r.min(),
            reward_max=r.max(),
            step=last_done_step,
            **sma.rs
        )
        print_func('-' * 40, out_time=True)
        print_func(f'Episode: {episode:3d} | step: {step:4d} | last_done_step {last_done_step:4d} | rewards: {arrprint(r, 3)}')

        if add_noise2buffer and episode % add_noise2buffer_episode_interval == 0:
            gym_random_sample(env, model, steps=add_noise2buffer_steps, print_func=print_func)

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
    tqdm_bar = trange(episodes_num, ncols=80)
    for _ in tqdm_bar:
        tqdm_bar.set_description('evaluating')
        model.reset()
        state[i] = env.reset()
        r = 0.
        step = 0
        while True:
            action = model.choose_action(s=state[0], visual_s=state[1], evaluation=True)
            state[i], reward, done, info = env.step(action)
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


def gym_random_sample(env, model, steps, print_func):
    i, state, new_state = init_variables(env)
    state[i] = env.reset()

    tqdm_bar = trange(steps, ncols=80)
    for _ in tqdm_bar:
        tqdm_bar.set_description('adding noise')
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
        state[i] = correct_new_state
    print_func('Noise added complete.')


def gym_evaluate(env, model, max_step_per_episode, max_eval_episode, print_func):
    i, state, _ = init_variables(env)
    total_r = np.zeros(env.n)
    total_steps = np.zeros(env.n)

    tqdm_bar = trange(0, max_eval_episode, env.n, ncols=80)
    for _ in tqdm_bar:
        tqdm_bar.set_description('evaluating')
        model.reset()
        state[i] = env.reset()
        dones_flag = np.full(env.n, False)
        steps = np.zeros(env.n)
        r = np.zeros(env.n)
        while True:
            r_tem = np.zeros(env.n)
            action = model.choose_action(s=state[0], visual_s=state[1], evaluation=True)  # In the future, this method can be combined with choose_action
            state[i], reward, done, info = env.step(action)
            model.partial_reset(done)
            unfinished_index = np.where(dones_flag == False)
            dones_flag += done
            r_tem[unfinished_index] = reward[unfinished_index]
            steps[unfinished_index] += 1
            r += r_tem
            if all(dones_flag) or any(steps >= max_step_per_episode):
                break
        total_r += r
        total_steps += steps
    average_r = total_r.mean() / tqdm_bar.total
    average_step = int(total_steps.mean() / tqdm_bar.total)
    solved = True if average_r >= env.reward_threshold else False
    print_func(f'evaluate number: {max_eval_episode:3d} | average step: {average_step} | average reward: {average_r} | SOLVED: {solved}')
    print_func('----------------------------------------------------------------------------------------------------------------------------')


def gym_no_op(env, model, print_func, pre_fill_steps, prefill_choose):
    assert isinstance(pre_fill_steps, int) and pre_fill_steps >= 0, 'no_op.steps must have type of int and larger than/equal 0'

    i, state, new_state = init_variables(env)
    model.reset()
    state[i] = env.reset()

    tqdm_bar = trange(0, pre_fill_steps, env.n, ncols=80)
    for _ in tqdm_bar:
        tqdm_bar.set_description('Pre-filling')

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


def gym_inference(env, model):
    i, state, _ = init_variables(env)
    model.reset()
    while True:
        step = 0
        state[i] = env.reset()
        while True:
            logger.info(f'step: {step}')
            env.render(record=False)
            action = model.choose_action(s=state[0], visual_s=state[1], evaluation=True)
            step += 1
            state[i], reward, done, info, correct_new_state = env.step(action)
            model.partial_reset(done)
            if done[0]:
                logger.info(f'done: {done[0]}, reward: {reward[0]}')
                step = 0
            state[i] = correct_new_state
