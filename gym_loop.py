import numpy as np
from utils.np_utils import SMA, arrprint


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
    return i, [np.array([[]] * env.n), np.array([[]] * env.n)], [np.array([[]] * env.n), np.array([[]] * env.n)]


class Loop(object):

    @staticmethod
    def train(env, gym_model, begin_episode, save_frequency, max_step, max_episode,
              eval_while_train, max_eval_episode, render, render_episode, policy_mode):
        """
        Inputs:
            env:                gym environment
            gym_model:          algorithm model
            begin_episode:      initial episode
            save_frequency:     how often to save checkpoints
            max_step:           maximum number of steps in an episode
            max_episode:        maximum number of episodes in this training task
            render:             specify whether render the env or not
            render_episode:     if 'render' is false, specify from which episode to render the env
        """
        i, state, new_state = init_variables(env)
        sma = SMA(100)
        for episode in range(begin_episode, max_episode):
            state[i] = env.reset()
            dones_flag = np.full(env.n, False)
            step = 0
            r = np.zeros(env.n)
            last_done_step = -1
            while True:
                step += 1
                r_tem = np.zeros(env.n)
                if render or episode > render_episode:
                    env.render()
                action = gym_model.choose_action(s=state[0], visual_s=state[1])
                new_state[i], reward, done, info = env.step(action)
                unfinished_index = np.where(dones_flag == False)[0]
                dones_flag += done
                r_tem[unfinished_index] = reward[unfinished_index]
                r += r_tem
                gym_model.store_data(
                    s=state[0],
                    visual_s=state[1],
                    a=action,
                    r=reward,
                    s_=new_state[0],
                    visual_s_=new_state[1],
                    done=done
                )

                if policy_mode == 'off-policy':
                    gym_model.learn(episode=episode, step=1)
                if all(dones_flag):
                    if last_done_step == -1:
                        last_done_step = step
                    if policy_mode == 'off-policy':
                        break

                if step >= max_step:
                    break

                if len(env.dones_index):    # 判断是否有线程中的环境需要局部reset
                    new_state[i][env.dones_index] = env.partial_reset()
                state[i] = new_state[i]

            sma.update(r)
            if policy_mode == 'on-policy':
                gym_model.learn(episode=episode, step=step)
            gym_model.writer_summary(
                episode,
                reward_mean=r.mean(),
                reward_min=r.min(),
                reward_max=r.max(),
                step=last_done_step,
                **sma.rs
            )
            print('-' * 40)
            print(f'Episode: {episode:3d} | step: {step:4d} | last_done_step {last_done_step:4d} | rewards: {arrprint(r, 3)}')
            if episode % save_frequency == 0:
                gym_model.save_checkpoint(episode)

            if eval_while_train and env.reward_threshold is not None:
                if r.max() >= env.reward_threshold:
                    ave_r, ave_step = Loop.evaluate(env, gym_model, max_step, max_eval_episode)
                    solved = True if ave_r >= env.reward_threshold else False
                    print(f'-------------------------------------------Evaluate episode: {episode:3d}--------------------------------------------------')
                    print(f'evaluate number: {max_eval_episode:3d} | average step: {ave_step} | average reward: {ave_r} | SOLVED: {solved}')
                    print('----------------------------------------------------------------------------------------------------------------------------')

    @staticmethod
    def evaluate(env, gym_model, max_step, max_eval_episode):
        i, state, _ = init_variables(env)
        total_r = np.zeros(env.n)
        total_steps = np.zeros(env.n)
        episodes = max_eval_episode // env.n
        for _ in range(episodes):
            state[i] = env.reset()
            dones_flag = np.full(env.n, False)
            steps = np.zeros(env.n)
            r = np.zeros(env.n)
            while True:
                r_tem = np.zeros(env.n)
                action = gym_model.choose_action(s=state[0], visual_s=state[1], evaluation=True)  # In the future, this method can be combined with choose_action
                state[i], reward, done, info = env.step(action)
                unfinished_index = np.where(dones_flag == False)
                dones_flag += done
                r_tem[unfinished_index] = reward[unfinished_index]
                steps[unfinished_index] += 1
                r += r_tem
                if all(dones_flag) or any(steps >= max_step):
                    break
            total_r += r
            total_steps += steps
        average_r = total_r.mean() / episodes
        average_step = int(total_steps.mean() / episodes)
        return average_r, average_step

    @staticmethod
    def inference(env, gym_model):
        """
        inference mode. algorithm model will not be train, only used to show agents' behavior
        """
        i, state, _ = init_variables(env)
        while True:
            state[i] = env.reset()
            while True:
                env.render()
                action = gym_model.choose_action(s=state[0], visual_s=state[1], evaluation=True)
                state[i], reward, done, info = env.step(action)
                if len(env.dones_index):    # 判断是否有线程中的环境需要局部reset
                    state[i][env.dones_index] = env.partial_reset()

    @staticmethod
    def no_op(env, gym_model, steps, choose=False):
        assert isinstance(steps, int) and steps >= 0, 'no_op.steps must have type of int and larger than/equal 0'
        i, state, new_state = init_variables(env)

        state[i] = env.reset()

        steps = steps // env.n + 1

        for step in range(steps):
            print(f'no op step {step}')
            if choose:
                action = gym_model.choose_action(s=state[0], visual_s=state[1])
            else:
                action = env.sample_actions()
            new_state[i], reward, done, info = env.step(action)
            gym_model.no_op_store(
                s=state[0],
                visual_s=state[1],
                a=action,
                r=reward,
                s_=new_state[0],
                visual_s_=new_state[1],
                done=done
            )
            if len(env.dones_index):    # 判断是否有线程中的环境需要局部reset
                new_state[i][env.dones_index] = env.partial_reset()
            state[i] = new_state[i]
