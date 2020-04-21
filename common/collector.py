import numpy as np

class GymCollector(object):

    def __init__(self):
        pass

    @staticmethod
    def run_batch_trajectory(env, model, steps=None):
        n = env.n
        i = 1 if env.obs_type == 'visual' else 0
        state = [np.array([[]] * n), np.array([[]] * n)]
        new_state = [np.array([[]] * n), np.array([[]] * n)]
        trajectories = [[] * n]
        
        model.reset()
        s = env.reset()
        dones_flag = np.full(n, False)

        while True:
            action = model.choose_action(s=state[0], visual_s=state[1])
            new_state[i], reward, done, info, correct_new_state = env.step(action)
            model.partial_reset(done)
            unfinished_index = np.where(dones_flag == False)[0]
            for i in unfinished_index:
                trajectories[i].append(
                    (state[0][i], state[1][i], action[i], reward[i], new_state[0][i], new_action[1][i], done[i])
                )
            state[i] = correct_new_state
            dones_flag += done
            if all(dones_flag):
                break

        return trajectories


class UnityCollector(object):

    def __init__(self):
        pass

    @staticmethod
    def run_batch_trajectory(env, model, steps):
        pass

    