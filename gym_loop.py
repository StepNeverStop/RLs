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
        dim = [obs_space.n] if type(obs_space.n) == int else list(obs_space.n)
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
    i = 1 if len(env.observation_space.shape) == 3 else 0
    mu, sigma = get_action_normalize_factor(env.action_space, action_type)
    if n == 2:
        return i, mu, sigma, [[np.array([[]] * env.n)] * 2] * 2
    else:
        return i, mu, sigma, [np.array([[]] * env.n)] * 2


class Loop(object):

    @staticmethod
    def train(env, gym_model, action_type, begin_episode, save_frequency, max_step, max_episode, render, render_episode, train_mode):
        i, mu, sigma, [state, new_state] = init_variables(env, action_type, 2)
        for episode in range(begin_episode, max_episode):
            obs = env.reset()
            state[i] = maybe_one_hot(obs, env.observation_space, env.n)
            step = 0
            r = np.zeros(env.n)
            while True:
                step += 1
                if render or episode > render_episode:
                    env.render()
                action = gym_model.choose_action(s=state[0], visual_s=state[1])
                obs, reward, done, info = env.step(action * sigma + mu)
                new_state[i] = maybe_one_hot(obs, env.observation_space, env.n)
                r += reward
                gym_model.store_data(
                    s=state[0],
                    visual_s=state[1],
                    a=action,
                    r=reward,
                    s_=new_state[0],
                    visual_s_=new_state[1],
                    done=done
                )
                state[i] = new_state[i]
                if train_mode == 'perStep':
                    gym_model.learn(episode)

                if all(done) or step > max_step:
                    break

            if train_mode == 'perEpisode':
                gym_model.learn(episode)

            print(f'episode {episode} step {step} rewards {r}')
            gym_model.writer_summary(
                episode,
                total_reward=r.mean(),
                step=step
            )
            if episode % save_frequency == 0:
                gym_model.save_checkpoint(episode)

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
                action = gym_model.choose_action(s=state[0], visual_s=state[1])
                obs, reward, done, info = env.step(action * sigma + mu)
                state[i] = maybe_one_hot(obs, env.observation_space, env.n)

    @staticmethod
    def no_op(env, gym_model, action_type, steps):
        assert type(steps) == int and steps >= 0
        i, mu, sigma, [state, new_state] = init_variables(env, action_type, 2)

        obs = env.reset()
        state[i] = maybe_one_hot(obs, env.observation_space, env.n)

        if action_type == 'continuous':
            action = np.zeros((env.n,) + env.action_space.shape, dtype=np.int32)
        else:
            action = np.zeros((env.n,), dtype=np.int32)

        for step in range(steps):
            print(f'no op step {step}')
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
