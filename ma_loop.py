import numpy as np


class Loop(object):

    @staticmethod
    def train(env, brain_names, models, data, begin_episode, save_frequency, reset_config, max_step, max_episode, sampler_manager, resampling_interval, policy_mode):
        assert policy_mode == 'off-policy', "multi-agents algorithms now support off-policy only."
        brains_num = len(brain_names)
        batch_size = data.batch_size
        agents_num = [0] * brains_num
        state = [0] * brains_num
        action = [0] * brains_num
        new_action = [0] * brains_num
        next_action = [0] * brains_num
        reward = [0] * brains_num
        next_state = [0] * brains_num
        dones = [0] * brains_num

        dones_flag = [0] * brains_num
        rewards = [0] * brains_num

        for episode in range(begin_episode, max_episode):
            if episode % resampling_interval == 0:
                reset_config.update(sampler_manager.sample_all())
            obs = env.reset(config=reset_config, train_mode=True)
            for i, brain_name in enumerate(brain_names):
                agents_num[i] = len(obs[brain_name].agents)
                dones_flag[i] = np.zeros(agents_num[i])
                rewards[i] = np.zeros(agents_num[i])
            step = 0
            last_done_step = -1
            while True:
                step += 1
                for i, brain_name in enumerate(brain_names):
                    state[i] = obs[brain_name].vector_observations
                    action[i] = models[i].choose_action(s=state[i])
                actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(brain_names)}
                obs = env.step(vector_action=actions)

                for i, brain_name in enumerate(brain_names):
                    reward[i] = np.array(obs[brain_name].rewards)[:, np.newaxis]
                    next_state[i] = obs[brain_name].vector_observations
                    dones[i] = np.array(obs[brain_name].local_done)[:, np.newaxis]
                    dones_flag[i] += obs[brain_name].local_done
                    rewards[i] += np.array(obs[brain_name].rewards)

                s = [np.array(e) for e in zip(*state)]
                a = [np.array(e) for e in zip(*action)]
                r = [np.array(e) for e in zip(*reward)]
                s_ = [np.array(e) for e in zip(*next_state)]
                done = [np.array(e) for e in zip(*dones)]
                data.add(s, a, r, s_, done)
                s, a, r, s_, done = data.sample()
                for i, brain_name in enumerate(brain_names):
                    next_action[i] = models[i].get_target_action(s=s_[:, i])
                    new_action[i] = models[i].choose_inference_action(s=s[:, i])
                a_ = np.array([np.array(e) for e in zip(*next_action)])
                if policy_mode == 'off-policy':
                    for i in range(brains_num):
                        models[i].learn(
                            episode=episode,
                            ap=np.array([np.array(e) for e in zip(*next_action[:i])]).reshape(batch_size, -1) if i != 0 else np.zeros((batch_size, 0)),
                            al=np.array([np.array(e) for e in zip(*next_action[-(brains_num - i - 1):])]).reshape(batch_size, -1) if brains_num - i != 1 else np.zeros((batch_size, 0)),
                            ss=s.reshape(batch_size, -1),
                            ss_=s_.reshape(batch_size, -1),
                            aa=a.reshape(batch_size, -1),
                            aa_=a_.reshape(batch_size, -1),
                            s=s[:, i],
                            r=r[:, i]
                        )

                if all([all(dones_flag[i]) for i in range(brains_num)]):
                    last_done_step = step
                    if policy_mode == 'off-policy':
                        break

                if step >= max_step:
                    break

            # if train_mode == 'perEpisode':
            #     for i in range(brains_num):
            #         models[i].learn(episode)

            for i in range(brains_num):
                models[i].writer_summary(
                    episode,
                    total_reward=rewards[i].mean(),
                    step=step
                )
            print(f'episode {episode:3d} step {step:4d} last_done_step {last_done_step:4d}')
            if episode % save_frequency == 0:
                for i in range(brains_num):
                    models[i].save_checkpoint(episode)

    @staticmethod
    def no_op(env, brain_names, models, data, brains, steps, choose=False, **kwargs):
        assert type(steps) == int, 'multi-agent no_op.steps must have type of int'
        if steps < data.batch_size:
            steps = data.batch_size
        brains_num = len(brain_names)
        agents_num = [0] * brains_num
        state = [0] * brains_num
        action = [0] * brains_num
        reward = [0] * brains_num
        next_state = [0] * brains_num
        dones = [0] * brains_num
        obs = env.reset(train_mode=False)

        for i, brain_name in enumerate(brain_names):
            agents_num[i] = len(obs[brain_name].agents)
            if brains[brain_name].vector_action_space_type == 'continuous':
                action[i] = np.zeros((agents_num[i], brains[brain_name].vector_action_space_size[0]), dtype=np.int32)
            else:
                action[i] = np.zeros((agents_num[i], len(brains[brain_name].vector_action_space_size)), dtype=np.int32)
                
        a = [np.array(e) for e in zip(*action)]
        for step in range(steps):
            print(f'no op step {step}')
            for i, brain_name in enumerate(brain_names):
                state[i] = obs[brain_name].vector_observations
                if choose:
                    action[i] = models[i].choose_action(s=state[i])
            actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(brain_names)}
            obs = env.step(vector_action=actions)
            for i, brain_name in enumerate(brain_names):
                reward[i] = np.array(obs[brain_name].rewards)[:, np.newaxis]
                next_state[i] = obs[brain_name].vector_observations
                dones[i] = np.array(obs[brain_name].local_done)[:, np.newaxis]
            s = [np.array(e) for e in zip(*state)]
            a = [np.array(e) for e in zip(*action)]
            r = [np.array(e) for e in zip(*reward)]
            s_ = [np.array(e) for e in zip(*next_state)]
            done = [np.array(e) for e in zip(*dones)]
            data.add(s, a, r, s_, done)

    @staticmethod
    def inference(env, brain_names, models, reset_config, sampler_manager, resampling_interval):
        """
        inference mode. algorithm model will not be train, only used to show agents' behavior
        """
        brains_num = len(brain_names)
        state = [0] * brains_num
        action = [0] * brains_num
        while True:
            if np.random.uniform() < 0.2:   # the environment has probability below 0.2 to change its parameters while running in the inference mode.
                reset_config.update(sampler_manager.sample_all())
            obs = env.reset(config=reset_config, train_mode=False)
            while True:
                for i, brain_name in enumerate(brain_names):
                    state[i] = obs[brain_name].vector_observations
                    action[i] = models[i].choose_inference_action(s=state[i])
                actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(brain_names)}
                obs = env.step(vector_action=actions)
