import sys
import numpy as np


class Loop(object):

    @staticmethod
    def train_OnPolicy(env, brain_names, models, begin_episode, save_frequency, reset_config, max_step, max_episode):
        brains_num = len(brain_names)
        state = [0] * brains_num
        action = [0] * brains_num
        dones_flag = [0] * brains_num
        agents_num = [0] * brains_num
        rewards = [0] * brains_num
        for episode in range(begin_episode, max_episode):
            obs = env.reset(config=reset_config, train_mode=True)
            for i, brain_name in enumerate(brain_names):
                agents_num[i] = len(obs[brain_name].agents)
                dones_flag[i] = np.zeros(agents_num[i])
                rewards[i] = np.zeros(agents_num[i])

                ss = []
                for j in range(agents_num[i]):
                    s = []
                    for k in range(models[i].visual_sources):
                        s.append(obs[brain_name].visual_observations[k][j])
                    ss.append([obs[brain_name].vector_observations[j], np.array(s)])
                state[i] = ss
            step = 0
            while True:
                step += 1
                for i, brain_name in enumerate(brain_names):
                    action[i] = models[i].choose_action(s=state[i])
                actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(brain_names)}
                obs = env.step(vector_action=actions)

                for i, brain_name in enumerate(brain_names):
                    dones_flag[i] += obs[brain_name].local_done
                    ss = []
                    for j in range(agents_num[i]):
                        s = []
                        for k in range(models[i].visual_sources):
                            s.append(obs[brain_name].visual_observations[k][j])
                        ss.append([obs[brain_name].vector_observations[j], np.array(s)])
                    models[i].store_data(
                        s=state[i],
                        a=action[i],
                        r=np.array(obs[brain_name].rewards),
                        s_=ss,
                        done=np.array(obs[brain_name].local_done)
                    )
                    state[i] = ss
                    rewards[i] += np.array(obs[brain_name].rewards)
                if all([all(dones_flag[i]) for i in range(brains_num)]) or step > max_step:
                    for i in range(brains_num):
                        models[i].learn(episode)
                        models[i].writer_summary(
                            episode,
                            total_reward=rewards[i].mean(),
                        )
                    break
            print(f'episode {episode} step {step}')
            if episode % save_frequency == 0:
                for i in range(brains_num):
                    models[i].save_checkpoint(episode)

    @staticmethod
    def train_OffPolicy(env, brain_names, models, begin_episode, save_frequency, reset_config, max_step, max_episode):
        brains_num = len(brain_names)
        state = [0] * brains_num
        action = [0] * brains_num
        dones_flag = [0] * brains_num
        agents_num = [0] * brains_num
        rewards = [0] * brains_num

        for episode in range(begin_episode, max_episode):
            obs = env.reset(config=reset_config, train_mode=True)
            for i, brain_name in enumerate(brain_names):
                agents_num[i] = len(obs[brain_name].agents)
                dones_flag[i] = np.zeros(agents_num[i])
                rewards[i] = np.zeros(agents_num[i])
                
                ss = []
                for j in range(agents_num[i]):
                    s = []
                    for k in range(models[i].visual_sources):
                        s.append(obs[brain_name].visual_observations[k][j])
                    ss.append([obs[brain_name].vector_observations[j], np.array(s)])
                state[i] = ss
            step = 0
            while True:
                step += 1
                for i, brain_name in enumerate(brain_names):
                    action[i] = models[i].choose_action(s=state[i])
                actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(brain_names)}
                obs = env.step(vector_action=actions)
                for i, brain_name in enumerate(brain_names):
                    dones_flag[i] += obs[brain_name].local_done
                    ss = []
                    for j in range(agents_num[i]):
                        s = []
                        for k in range(models[i].visual_sources):
                            s.append(obs[brain_name].visual_observations[k][j])
                        ss.append([obs[brain_name].vector_observations[j], np.array(s)])
                    models[i].store_data(
                        s=state[i],
                        a=action[i],
                        r=np.array(obs[brain_name].rewards)[:, np.newaxis],
                        s_=ss,
                        done=np.array(obs[brain_name].local_done)[:, np.newaxis]
                    )
                    state[i] = ss
                    models[i].learn(episode)
                    rewards[i] += np.array(obs[brain_name].rewards)
                if all([all(dones_flag[i]) for i in range(brains_num)]) or step > max_step:
                    break
            print(f'episode {episode} step {step}')
            for i in range(brains_num):
                models[i].writer_summary(
                    episode,
                    total_reward=rewards[i].mean()
                )
            if episode % save_frequency == 0:
                for i in range(brains_num):
                    models[i].save_checkpoint(episode)

    @staticmethod
    def inference(env, brain_names, models, reset_config):
        brains_num = len(brain_names)
        state = [0] * brains_num
        action = [0] * brains_num
        agents_num = [0] * brains_num
        while True:
            obs = env.reset(config=reset_config, train_mode=False)
            for i, brain_name in enumerate(brain_names):
                agents_num[i] = len(obs[brain_name].agents)
            while True:
                for i, brain_name in enumerate(brain_names):
                    ss = []
                    for j in range(agents_num[i]):
                        s = []
                        for k in range(models[i].visual_sources):
                            s.append(obs[brain_name].visual_observations[k][j])
                        ss.append([obs[brain_name].vector_observations[j], np.array(s)])
                    state[i] = ss
                    action[i] = models[i].choose_inference_action(s=state[i])
                actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(brain_names)}
                obs = env.step(vector_action=actions)
