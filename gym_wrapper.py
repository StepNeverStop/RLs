import gym
import numpy as np
import threading

class MyThread(threading.Thread):

    def __init__(self,func,args=()):
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
        self.a_type = 'discrete' if type(self.envs[0].action_space) == gym.spaces.discrete.Discrete else 'continuous'
        self.action_space = self.envs[0].action_space
        print(self.action_space)
    
    def render(self):
        self.envs[0].render()
    
    def close(self):
        [env.close() for env in self.envs]
    
    def reset(self):
        threadpool = []
        for i in range(self.n):
            th = MyThread(self.envs[i].reset, args=())
            threadpool.append(th)
        for th in threadpool:
            th.start()
        for th in threadpool:
            threading.Thread.join(th)
        if self.obs_type == 'visual':
            return [threadpool[i].get_result()[np.newaxis, :] for i in range(self.n)]
        else:
            return [threadpool[i].get_result() for i in range(self.n)]

    def step(self, actions):
        if self.a_type == 'discrete':
            actions = np.squeeze(actions)
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
        return [np.array(e) for e in zip(*results)]

    