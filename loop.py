import numpy as np


def get_visual_input(n, cameras, brain_obs):
    '''
    inputs:
        n: agents number
        cameras: camera number
        brain_obs: observations of specified brain, include visual and vector observation.
    output:
        [vector_information, [visual_info0, visual_info1, visual_info2, ...]]
    '''
    ss = []
    for j in range(n):
        s = []
        for k in range(cameras):
            s.append(brain_obs.visual_observations[k][j])
        ss.append(np.array(s))
    return np.array(ss)


class Loop(object):

    @staticmethod
    def train(env, brain_names, models, begin_episode, save_frequency, reset_config, max_step, max_episode, train_mode, sampler_manager, resampling_interval):
        """
        Train loop. Execute until episode reaches its maximum or press 'ctrl+c' artificially.
        Inputs:
            env:                    Environment for interaction.
            models:                 all models for this trianing task.
            save_frequency:         how often to save checkpoints.
            reset_config:           configuration to reset for Unity environment.
            max_step:               maximum number of steps for an episode.
            train_mode:             train or inference.
            sampler_manager:        sampler configuration parameters for 'reset_config'.
            resampling_interval:    how often to resample parameters for env reset.
        Variables:
            brain_names:    a list of brain names set in Unity.
            state: store    a list of states for each brain. each item contain a list of states for each agents that controlled by the same brain.
            visual_state:   store a list of visual state information for each brain.
            action:         store a list of actions for each brain.
            dones_flag:     store a list of 'done' for each brain. use for judge whether an episode is finished for every agents.
            agents_num:     use to record 'number' of agents for each brain.
            rewards:        use to record rewards of agents for each brain.
        """
        brains_num = len(brain_names)
        state = [0] * brains_num
        visual_state = [0] * brains_num
        action = [0] * brains_num
        dones_flag = [0] * brains_num
        agents_num = [0] * brains_num
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
            while True:
                step += 1
                for i, brain_name in enumerate(brain_names):
                    state[i] = obs[brain_name].vector_observations
                    visual_state[i] = get_visual_input(agents_num[i], models[i].visual_sources, obs[brain_name])
                    action[i] = models[i].choose_action(s=state[i], visual_s=visual_state[i])
                actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(brain_names)}
                obs = env.step(vector_action=actions)

                for i, brain_name in enumerate(brain_names):
                    dones_flag[i] += obs[brain_name].local_done
                    next_state = obs[brain_name].vector_observations
                    next_visual_state = get_visual_input(agents_num[i], models[i].visual_sources, obs[brain_name])
                    models[i].store_data(
                        s=state[i],
                        visual_s=visual_state[i],
                        a=action[i],
                        r=np.array(obs[brain_name].rewards),
                        s_=next_state,
                        visual_s_=next_visual_state,
                        done=np.array(obs[brain_name].local_done)
                    )
                    rewards[i] += np.array(obs[brain_name].rewards)

                if train_mode == 'perStep':
                    for i in range(brains_num):
                        models[i].learn(episode)

                if all([all(dones_flag[i]) for i in range(brains_num)]) or step > max_step:
                    break

            if train_mode == 'perEpisode':
                for i in range(brains_num):
                    models[i].learn(episode)

            for i in range(brains_num):
                models[i].writer_summary(
                    episode,
                    total_reward=rewards[i].mean(),
                    step=step
                )
            print(f'episode {episode} step {step}')
            if episode % save_frequency == 0:
                for i in range(brains_num):
                    models[i].save_checkpoint(episode)

    @staticmethod
    def inference(env, brain_names, models, reset_config, sampler_manager, resampling_interval):
        """
        inference mode. algorithm model will not be train, only used to show agents' behavior
        """
        brains_num = len(brain_names)
        state = [0] * brains_num
        visual_state = [0] * brains_num
        action = [0] * brains_num
        agents_num = [0] * brains_num
        while True:
            if np.random.uniform() < 0.2:   # the environment has probability below 0.2 to change its parameters while running in the inference mode.
                reset_config.update(sampler_manager.sample_all())
            obs = env.reset(config=reset_config, train_mode=False)
            for i, brain_name in enumerate(brain_names):
                agents_num[i] = len(obs[brain_name].agents)
            while True:
                for i, brain_name in enumerate(brain_names):
                    state[i] = obs[brain_name].vector_observations
                    visual_state[i] = get_visual_input(agents_num[i], models[i].visual_sources, obs[brain_name])
                    action[i] = models[i].choose_inference_action(s=state[i], visual_s=visual_state[i])
                actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(brain_names)}
                obs = env.step(vector_action=actions)

    @staticmethod
    def no_op(env, brain_names, models, brains, steps):
        '''
        Interact with the environment but do not perform actions. Prepopulate the ReplayBuffer.
        Make sure steps is greater than n-step if using any n-step ReplayBuffer.
        '''
        assert type(steps) == int and steps >= 0, 'no_op.steps must have type of int and larger than/equal 0'
        brains_num = len(brain_names)
        state = [0] * brains_num
        visual_state = [0] * brains_num
        agents_num = [0] * brains_num
        action = [0] * brains_num
        obs = env.reset(train_mode=False)

        for i, brain_name in enumerate(brain_names):
            agents_num[i] = len(obs[brain_name].agents)
            if brains[brain_name].vector_action_space_type == 'continuous':
                action[i] = np.zeros((agents_num[i], brains[brain_name].vector_action_space_size[0]), dtype=np.int32)
            else:
                action[i] = np.zeros((agents_num[i], len(brains[brain_name].vector_action_space_size)), dtype=np.int32)
        actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(brain_names)}

        for step in range(steps):
            print(f'no op step {step}')
            for i, brain_name in enumerate(brain_names):
                state[i] = obs[brain_name].vector_observations
                visual_state[i] = get_visual_input(agents_num[i], models[i].visual_sources, obs[brain_name])
            obs = env.step(vector_action=actions)
            for i, brain_name in enumerate(brain_names):
                next_state = obs[brain_name].vector_observations
                next_visual_state = get_visual_input(agents_num[i], models[i].visual_sources, obs[brain_name])
                models[i].no_op_store(
                    s=state[i],
                    visual_s=visual_state[i],
                    a=action[i],
                    r=np.array(obs[brain_name].rewards),
                    s_=next_state,
                    visual_s_=next_visual_state,
                    done=np.array(obs[brain_name].local_done)
                )
