import numpy as np
from Algorithms.tf2algos.base.policy import Policy
from utils.sth import sth
from utils.replay_buffer import ExperienceReplay, NStepExperienceReplay, PrioritizedExperienceReplay, NStepPrioritizedExperienceReplay, er_config


class Off_Policy(Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 action_type,
                 **kwargs):
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim_or_list=a_dim_or_list,
            action_type=action_type,
            **kwargs)
        self.batch_size = int(kwargs.get('batch_size', 128))
        self.buffer_size = int(kwargs.get('buffer_size', 10000))
        self.use_priority = kwargs.get('use_priority', False)
        self.n_step = kwargs.get('n_step', False)
        self.init_data_memory()

    def init_data_memory(self):
        if self.use_priority:
            if self.n_step:
                print('N-Step PER')
                self.data = NStepPrioritizedExperienceReplay(self.batch_size,
                                                                self.buffer_size,
                                                                max_episode=self.max_episode,
                                                                gamma=self.gamma,
                                                                alpha=er_config['nper_config']['alpha'],
                                                                beta=er_config['nper_config']['beta'],
                                                                epsilon=er_config['nper_config']['epsilon'],
                                                                agents_num=er_config['nper_config']['max_agents'],
                                                                n=er_config['nper_config']['n'],
                                                                global_v=er_config['nper_config']['global_v'])
            else:
                print('PER')
                self.data = PrioritizedExperienceReplay(self.batch_size,
                                                        self.buffer_size,
                                                        max_episode=self.max_episode,
                                                        alpha=er_config['per_config']['alpha'],
                                                        beta=er_config['per_config']['beta'],
                                                        epsilon=er_config['per_config']['epsilon'],
                                                        global_v=er_config['nper_config']['global_v'])
        else:
            if self.n_step:
                print('N-Step ER')
                self.data = NStepExperienceReplay(self.batch_size,
                                                    self.buffer_size,
                                                    gamma=self.gamma,
                                                    agents_num=er_config['ner_config']['max_agents'],
                                                    n=er_config['ner_config']['n'])
            else:
                print('ER')
                self.data = ExperienceReplay(self.batch_size, self.buffer_size)


    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        """
        for off-policy training, use this function to store <s, a, r, s_, done> into ReplayBuffer.
        """
        assert isinstance(a, np.ndarray), "store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store need done type is np.ndarray"
        if not self.action_type == 'continuous':
            a = sth.action_index2one_hot(a, self.a_dim_or_list)
        self.data.add(
            s,
            visual_s,
            a,
            r[:, np.newaxis],
            s_,
            visual_s_,
            done[:, np.newaxis]
        )

    def no_op_store(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "no_op_store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "no_op_store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "no_op_store need done type is np.ndarray"
        if not self.action_type == 'continuous':
            a = sth.action_index2one_hot(a, self.a_dim_or_list)
        self.data.add(
            s,
            visual_s,
            a,
            r[:, np.newaxis],
            s_,
            visual_s_,
            done[:, np.newaxis]
        )
