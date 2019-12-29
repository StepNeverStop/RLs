
import numpy as np
from utils.sampler import create_sampler_manager


class BasicWrapper:
    def __init__(self, env):
        self.env = env

    def step(self, **kwargs):
        return self.env.step(**kwargs)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def close(self):
        self.env.close()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)


class InfoWrapper(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.brains = self.env.brains
        self.brain_names = self.env.external_brain_names
        self.brain_num = len(self.brain_names)

        self.visual_sources = [self.brains[b].number_visual_observations for b in self.brain_names]
        self.visual_resolutions = []
        for b in self.brain_names:
            if self.brains[b].number_visual_observations:
                self.visual_resolutions.append([
                    self.brains[b].camera_resolutions[0]['height'],
                    self.brains[b].camera_resolutions[0]['width'],
                    1 if self.brains[b].camera_resolutions[0]['blackAndWhite'] else 3
                ])
            else:
                self.visual_resolutions.append([])

        self.s_dim = [self.brains[b].vector_observation_space_size * self.brains[b].num_stacked_vector_observations
                      for b in self.brain_names]
        self.a_dim_or_list = [self.brains[b].vector_action_space_size for b in self.brain_names]
        self.is_continuous = [True if self.brains[b].vector_action_space_type == 'continuous' else False
                              for b in self.brain_names]
        obs = self.env.reset()
        self.brain_agents = [len(obs[brain_name].agents) for brain_name in self.brain_names]

    def random_action(self):
        '''
        choose random action for each brain and each agent.
        continuous: [-1, 1]
        discrete: [0-max, 0-max, ...] i.e. action dim = [2, 3] => action range from [0, 0] to [1, 2].
        '''
        actions = []
        for i in range(self.brain_num):
            if self.is_continuous[i]:
                actions.append(
                    np.random.random((self.brain_agents[i], self.a_dim_or_list[i][0])) * 2 - 1
                )
            else:
                actions.append(
                    np.random.randint(self.a_dim_or_list[i], size=(self.brain_agents[i], len(self.a_dim_or_list[i])), dtype=np.int32)
                )
        return actions


class UnityReturnWrapper(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.splitByBrain(obs)

    def step(self, **kwargs):
        obs = self.env.step(**kwargs)
        return self.splitByBrain(obs)

    def splitByBrain(self, obs):
        vector = [obs[brain_name].vector_observations for brain_name in self.brain_names]
        visual = [self._get_visual_input(n=n, cameras=cameras, brain_obs=obs[brain_name])
                  for n, cameras, brain_name in zip(self.brain_agents, self.visual_sources, self.brain_names)]
        reward = [np.asarray(obs[brain_name].rewards) for brain_name in self.brain_names]
        done = [np.asarray(obs[brain_name].local_done) for brain_name in self.brain_names]
        return zip(vector, visual, reward, done)

    def _get_visual_input(self, n, cameras, brain_obs):
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


class SamplerWrapper(BasicWrapper):
    def __init__(self, env, env_args):
        super().__init__(env)
        self.reset_config = env_args['reset_config']
        self.train_mode = env_args['train_mode']
        self.sampler_manager, self.resample_interval = create_sampler_manager(env_args['sampler_path'], self.env.reset_parameters)
        self.episode = 0

    def reset(self):
        self.episode += 1
        if self.episode % self.resample_interval == 0:
            self.reset_config.update(self.sampler_manager.sample_all())
        obs = self.env.reset(config=self.reset_config, train_mode=self.train_mode)
        return obs
