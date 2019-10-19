import numpy as np


def get_action_normalize_factor(space, action_type):
    '''
    input: {'low': [-2, -3], 'high': [2, 6]}, 'continuous'
    return: [0, 1.5], [2, 4.5]
    '''
    if action_type == 'continuous':
        return (space.high + space.low) / 2, (space.high - space.low) / 2
    else:
        return 0, 1


def maybe_one_hot(obs, obs_space, n):
    """
    input: [[1, 0], [2, 1]], (3, 4), 2
    output: [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
             [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
    """
    if hasattr(obs_space, 'n'):
        obs = obs.reshape(n, -1)
        dim = [int(obs_space.n)] if type(obs_space.n) == int or type(obs_space.n) == np.int32 else list(obs_space.n)    # 在CliffWalking-v0环境其类型为numpy.int32
        multiplication_factor = dim[1:] + [1]
        n = np.array(dim).prod()
        ints = obs.dot(multiplication_factor)
        x = np.zeros([obs.shape[0], n])
        for i, j in enumerate(ints):
            x[i, j] = 1
        return x
    else:
        return obs


def init_variables(env, action_type, n):
    """
    inputs:
        env: Environment
        action_type: discrete or continuous
        n: number of state array
    outputs:
        i: specify which item of state should be modified
        mu: action bias
        sigma: action scale
    """
    i = 1 if len(env.observation_space.shape) == 3 else 0
    mu, sigma = get_action_normalize_factor(env.action_space, action_type)
    if n == 2:
        return i, mu, sigma, [[np.array([[]] * env.n)] * 2] * 2
    else:
        return i, mu, sigma, [np.array([[]] * env.n)] * 2


class Loop(object):

    @staticmethod
    def train(env, gym_model, action_type, begin_episode, save_frequency, max_step, max_episode, eval_while_train, max_eval_episode, render, render_episode, train_mode):
        """
        Inputs:
            env:                gym environment
            gym_model:          algorithm model
            action_type:        specify action type, discrete action space or continuous action space
            begin_episode:      initial episode
            save_frequency:     how often to save checkpoints
            max_step:           maximum number of steps in an episode
            max_episode:        maximum number of episodes in this training task
            render:             specify whether render the env or not
            render_episode:     if 'render' is false, specify from which episode to render the env
            train_mode:         perStep or perEpisode
        """
        i, mu, sigma, [state, new_state] = init_variables(env, action_type, 2)
        for episode in range(begin_episode, max_episode):
            obs = env.reset()
            state[i] = maybe_one_hot(obs, env.observation_space, env.n)
            dones_flag = np.full(env.n, False)
            step_max_of_all = 0
            r = np.zeros(env.n)
            while True:
                step_max_of_all += 1
                r_tem = np.zeros(env.n)
                if render or episode > render_episode:
                    env.render()
                action = gym_model.choose_action(s=state[0], visual_s=state[1])
                obs, reward, done, info = env.step(action * sigma + mu)
                unfinished_index = np.where(dones_flag == False)[0]
                dones_flag += done
                new_state[i] = maybe_one_hot(obs, env.observation_space, env.n)
                r_tem[unfinished_index] = reward[unfinished_index]
                r += r_tem
                gym_model.store_data(
                    s=state[0],
                    visual_s=state[1],
                    a=action,
                    r=reward.astype(np.float64),
                    s_=new_state[0],
                    visual_s_=new_state[1],
                    done=done
                )

                if train_mode == 'perStep':
                    gym_model.learn(episode)

                if all(dones_flag) or step_max_of_all >= max_step:
                    break
                
                if len(env.dones_index):    # 判断是否有线程中的环境需要局部reset
                    new_episode_states = maybe_one_hot(env.patial_reset(), env.observation_space, len(env.dones_index))
                    new_state[i][env.dones_index] = new_episode_states
                state[i] = new_state[i]

            if train_mode == 'perEpisode':
                gym_model.learn(episode)

            print(f'episode {episode} step_max_of_all {step_max_of_all} rewards {r}')
            gym_model.writer_summary(
                episode,
                total_reward=r.mean(),
                step=step_max_of_all
            )
            if episode % save_frequency == 0:
                gym_model.save_checkpoint(episode)

            if r.max() > 0 and eval_while_train:
                ave_r, ave_step = Loop.evaluate(env, gym_model, action_type, max_step, max_eval_episode)
                print('--------------------------------------------------------------------------------------------')
                print(f'eval episode {episode} evaluate number {max_eval_episode} average step {ave_step} average reward {ave_r}')
                print('--------------------------------------------------------------------------------------------')

    @staticmethod
    def evaluate(env, gym_model, action_type, max_step, max_eval_episode):
        i, mu, sigma, [state, _] = init_variables(env, action_type, 2)
        total_r = np.zeros(env.n)
        total_steps = np.zeros(env.n)
        episodes = max_eval_episode // env.n
        for _ in range(episodes):
            obs = env.reset()
            state[i] = maybe_one_hot(obs, env.observation_space, env.n)
            dones_flag = np.full(env.n, False)
            steps = np.zeros(env.n)
            r = np.zeros(env.n)
            while True:
                r_tem = np.zeros(env.n)
                action = gym_model.choose_inference_action(s=state[0], visual_s=state[1])
                obs, reward, done, info = env.step(action * sigma + mu)
                unfinished_index = np.where(dones_flag == False)
                dones_flag += done
                state[i] = maybe_one_hot(obs, env.observation_space, env.n)
                r_tem[unfinished_index] = reward[unfinished_index]
                steps[unfinished_index] += 1
                r += r_tem
                if all(dones_flag) or any(steps >= max_step):
                    break
            total_r += r
            total_steps += steps
        average_r = total_r.mean() / episodes
        average_step = total_steps.mean() / episodes
        return average_r, average_step

    @staticmethod
    def inference(env, gym_model, action_type):
        """
        inference mode. algorithm model will not be train, only used to show agents' behavior
        """
        i, mu, sigma, new_state = init_variables(env, action_type, 1)
        while True:
            obs = env.reset()
            state[i] = maybe_one_hot(obs, env.observation_space, env.n)
            while True:
                env.render()
                action = gym_model.choose_inference_action(s=state[0], visual_s=state[1])
                obs, reward, done, info = env.step(action * sigma + mu)
                state[i] = maybe_one_hot(obs, env.observation_space, env.n)

    @staticmethod
    def no_op(env, gym_model, action_type, steps, choose=False):
        assert type(steps) == int and steps >= 0, 'no_op.steps must have type of int and larger than/equal 0'
        i, mu, sigma, [state, new_state] = init_variables(env, action_type, 2)

        obs = env.reset()
        state[i] = maybe_one_hot(obs, env.observation_space, env.n)

        if action_type == 'continuous':
            action = np.zeros((env.n,) + env.action_space.shape, dtype=np.int32)
        else:
            tmp = (len(env.action_space),) if hasattr(env.action_space, '__len__') else ()
            action = np.zeros((env.n,) + tmp, dtype=np.int32)

        steps = steps // env.n + 1

        for step in range(steps):
            print(f'no op step {step}')
            if choose:
                action = gym_model.choose_action(s=state[0], visual_s=state[1])
            obs, reward, done, info = env.step(action * sigma + mu)
            new_state[i] = maybe_one_hot(obs, env.observation_space, env.n)
            gym_model.no_op_store(
                s=state[0],
                visual_s=state[1],
                a=action,
                r=reward,
                s_=new_state[0],
                visual_s_=new_state[1],
                done=done
            )
            state[i] = new_state[i]
