
from utils.sampler import create_sampler_manager


class UnityWrapper:
    def __init__(self, env, env_args):
        self.env = env
        self.reset_config = env_args['reset_config']
        self.train_mode = env_args['train_mode']
        self.sampler_manager, self.resample_interval = create_sampler_manager(env_args['sampler_path'], env.reset_parameters)
        self.episode = 0

        self.brains = env.brains
        self.brain_names = env.external_brain_names
        self.brain_num = len(self.brain_names)
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
        self.visual_sources = [self.brains[b].number_visual_observations for b in self.brain_names]
        self.s_dim = [self.brains[b].vector_observation_space_size * self.brains[b].num_stacked_vector_observations
                      for b in self.brain_names]
        self.a_dim_or_list = [self.brains[b].vector_action_space_size for b in self.brain_names]
        self.is_continuous = [True if self.brains[b].vector_action_space_type == 'continuous' else False
                              for b in self.brain_names]

        obs = self.env.reset()
        self.brain_agents = [len(obs[brain_name].agents) for brain_name in self.brain_names]

    def reset(self):
        self.episode += 1
        if self.episode % self.resample_interval == 0:
            self.reset_config.update(self.sampler_manager.sample_all())
        obs = self.env.reset(config=self.reset_config, train_mode=self.train_mode)
        return obs

    def step(self, **kwargs):
        return self.env.step(**kwargs)

    def close(self):
        self.env.close()
