import sys
import numpy as np


class Loop(object):
    @staticmethod
    def inference(env, brain_names, models, reset_config):
        while True:
            obs = env.reset(config=reset_config, train_mode=False)
            while True:
                for i, brain_name in enumerate(brain_names):
                    state[i] = obs[brain_name].vector_observations
                    action[i] = models[i].choose_inference_action(s=state[i])
                actions = {f'{brain_name}': action[i]
                           for i, brain_name in enumerate(brain_names)}
                obs = env.step(vector_action=actions)

    @staticmethod
    def train_OnPolicy(env, brain_names, models, begin_episode, save_frequency, reset_config):
        brains_num = len(brain_names)
        state = [0] * brains_num
        action = [0] * brains_num
        dones_flag = [0] * brains_num
        agents_num = [0] * brains_num

        for episode in range(begin_episode, 50000):
            obs = env.reset(config=reset_config, train_mode=True)
            for i, brain_name in enumerate(brain_names):
                agents_num[i] = len(obs[brain_name].agents)
                dones_flag[i] = np.zeros(agents_num[i])

            step = 0

            while True:
                step += 1

                for i, brain_name in enumerate(brain_names):
                    state[i] = obs[brain_name].vector_observations
                    action[i] = models[i].choose_action(s=state[i])

                actions = {f'{brain_name}': action[i]
                           for i, brain_name in enumerate(brain_names)}
                obs = env.step(vector_action=actions)

                for i, brain_name in enumerate(brain_names):
                    dones_flag[i] += obs[brain_name].local_done
                    models[i].store_data(
                        s=state[i],
                        a=action[i],
                        r=np.array(obs[brain_name].rewards),
                        s_=obs[brain_name].vector_observations,
                        done=np.array(obs[brain_name].local_done)
                    )
                if all([all(dones_flag[i]) for i in range(brains_num)]):
                    for i in range(brains_num):
                        models[i].learn()
                        models[i].writer_summary(episode)
                    break
            if episode % save_frequency == 0:
                for i in range(brains_num):
                    models[i].save_checkpoint(episode)
            print(f'episode {episode} step {step}')

    @staticmethod
    def train_OffPolicy(env, brain_names, models, begin_episode, save_frequency, reset_config):
        brains_num = len(brain_names)
        state = [0] * brains_num
        action = [0] * brains_num
        dones_flag = [0] * brains_num
        agents_num = [0] * brains_num

        for episode in range(begin_episode, 50000):
            obs = env.reset(config=reset_config, train_mode=True)
            for i, brain_name in enumerate(brain_names):
                agents_num[i] = len(obs[brain_name].agents)
                dones_flag[i] = np.zeros(agents_num[i])

            step = 0

            while True:
                step += 1

                for i, brain_name in enumerate(brain_names):
                    state[i] = obs[brain_name].vector_observations
                    action[i] = models[i].choose_action(s=state[i])

                actions = {f'{brain_name}': action[i]
                           for i, brain_name in enumerate(brain_names)}
                obs = env.step(vector_action=actions)

                for i, brain_name in enumerate(brain_names):
                    dones_flag[i] += obs[brain_name].local_done
                    models[i].store_data(
                        s=state[i],
                        a=action[i],
                        r=np.array(obs[brain_name].rewards)[:, np.newaxis],
                        s_=obs[brain_name].vector_observations,
                        done=np.array(obs[brain_name].local_done)[
                            :, np.newaxis]
                    )
                    models[i].learn()
                if all([all(dones_flag[i]) for i in range(brains_num)]):
                    break
            for i in range(brains_num):
                models[i].writer_summary(episode)
            if episode % save_frequency == 0:
                for i in range(brains_num):
                    models[i].save_checkpoint(episode)
            print(f'episode {episode} step {step}')
