import gym
import numpy as np
import threading


class MyThread(threading.Thread):

    def __init__(self, func, args=()):
        super().__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


class gym_envs(object):

    def __init__(self, gym_env_name, n):
        self.n = n
        self.envs = [gym.make(gym_env_name) for _ in range(self.n)]
        self.observation_space = self.envs[0].observation_space
        self.obs_type = 'visual' if len(self.observation_space.shape) == 3 else 'vector'
        self.reward_threshold = self.envs[0].env.spec.reward_threshold
        if type(self.envs[0].action_space) == gym.spaces.box.Box:
            self.a_type = 'continuous'
        elif type(self.envs[0].action_space) == gym.spaces.tuple.Tuple:
            self.a_type = 'Tuple(Discrete)'
        else:
            self.a_type = 'discrete'
        self.action_space = self.envs[0].action_space

    def render(self):
        self.envs[0].render()

    def close(self):
        [env.close() for env in self.envs]
    
    def sample_action(self):
        return np.array([env.action_space.sample() for env in self.envs])

    def reset(self):
        self.dones_index = []
        threadpool = []
        for i in range(self.n):
            th = MyThread(self.envs[i].reset, args=())
            threadpool.append(th)
        for th in threadpool:
            th.start()
        for th in threadpool:
            threading.Thread.join(th)
        if self.obs_type == 'visual':
            return np.array([threadpool[i].get_result()[np.newaxis, :] for i in range(self.n)])
        else:
            return np.array([threadpool[i].get_result() for i in range(self.n)])

    def step(self, actions):
        if self.a_type == 'discrete':
            actions = actions.reshape(-1,)
        elif self.a_type == 'Tuple(Discrete)':
            actions = actions.reshape(self.n, -1).tolist()
        threadpool = []
        for i in range(self.n):
            th = MyThread(self.envs[i].step, args=(actions[i], ))
            threadpool.append(th)
        for th in threadpool:
            th.start()
        for th in threadpool:
            threading.Thread.join(th)
        if self.obs_type == 'visual':
            results = [
                [threadpool[i].get_result()[0][np.newaxis, :], *threadpool[i].get_result()[1:]]
                for i in range(self.n)]
        else:
            results = [threadpool[i].get_result() for i in range(self.n)]
        obs, reward, done, info = [np.array(e) for e in zip(*results)]
        self.dones_index = np.where(done)[0]
        return obs, reward, done, info
    
    def patial_reset(self):
        threadpool = []
        for i in self.dones_index:
            th = MyThread(self.envs[i].reset, args=())
            threadpool.append(th)
        for th in threadpool:
            th.start()
        for th in threadpool:
            threading.Thread.join(th)
        if self.obs_type == 'visual':
            return np.array([threadpool[i].get_result()[np.newaxis, :] for i in range(self.dones_index.shape[0])])
        else:
            return np.array([threadpool[i].get_result() for i in range(self.dones_index.shape[0])])

