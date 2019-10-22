import numpy as np
import pandas as pd
import tensorflow as tf
from .base import Base
from utils.sth import sth
from utils.replay_buffer import ExperienceReplay, NStepExperienceReplay, PrioritizedExperienceReplay, NStepPrioritizedExperienceReplay, er_config


class Policy(Base):
    def __init__(self,
                 a_dim_or_list,
                 action_type,
                 base_dir,

                 s_dim,
                 visual_sources,
                 visual_resolution,
                 gamma,
                 max_episode,
                 policy_mode=None,
                 batch_size=1,
                 buffer_size=1,
                 use_priority=False,
                 n_step=False):
        super().__init__(
            a_dim_or_list=a_dim_or_list,
            action_type=action_type,
            base_dir=base_dir)
        self.s_dim = s_dim
        self.visual_sources = visual_sources
        self.visual_dim = [visual_sources, *visual_resolution] if visual_sources else [0]
        self.a_dim_or_list = a_dim_or_list
        self.gamma = gamma
        self.max_episode = max_episode
        self.policy_mode = policy_mode
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.use_priority = use_priority
        self.n_step = n_step
        self.init_data_memory()

    def init_data_memory(self):
        '''
        the biggest diffenernce between policy_modes(ON and OFF) is 'OFF' mode need raise the dimension
        of 'r' and 'done'.
        'ON' mode means program will call on_store function and use pandas dataframe to store data.
        'OFF' mode will call off_store function and use replay buffer to store data.
        '''
        if self.policy_mode == 'ON':
            self.data = pd.DataFrame(columns=['s', 'a', 'r', 'done'])
        elif self.policy_mode == 'OFF':
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
        else:
            raise Exception('Please specific a mode of policy!')

    def on_store(self, s, visual_s, a, r, s_, visual_s_, done):
        """
        for on-policy training, use this function to store <s, a, r, s_, done> into DataFrame of Pandas.
        """
        assert isinstance(a, np.ndarray), "on_store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "on_store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "on_store need done type is np.ndarray"
        if not self.action_type == 'continuous':
            a = sth.action_index2one_hot(a, self.a_dim_or_list)
        self.data = self.data.append({
            's': s.astype(np.float32),
            'visual_s': visual_s.astype(np.float32),
            'a': a.astype(np.float32),
            'r': r.astype(np.float32),
            's_': s_.astype(np.float32),
            'visual_s_': visual_s_.astype(np.float32),
            'done': done.astype(np.float32)
        }, ignore_index=True)

    def off_store(self, s, visual_s, a, r, s_, visual_s_, done):
        """
        for off-policy training, use this function to store <s, a, r, s_, done> into ReplayBuffer.
        """
        assert isinstance(a, np.ndarray), "off_store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "off_store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "off_store need done type is np.ndarray"
        if not self.action_type == 'continuous':
            a = sth.action_index2one_hot(a, self.a_dim_or_list)
        self.data.add(
            s.astype(np.float32), 
            visual_s.astype(np.float32), 
            a.astype(np.float32), 
            r.astype(np.float32), 
            s_.astype(np.float32), 
            visual_s_.astype(np.float32), 
            done.astype(np.float32)
            )

    def no_op_store(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "no_op_store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "no_op_store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "no_op_store need done type is np.ndarray"
        if self.policy_mode == 'OFF':
            if not self.action_type == 'continuous':
                a = sth.action_index2one_hot(a, self.a_dim_or_list)
            self.data.add(
                s.astype(np.float32), 
                visual_s.astype(np.float32), 
                a.astype(np.float32), 
                r[:, np.newaxis].astype(np.float32), 
                s_.astype(np.float32), 
                visual_s_.astype(np.float32), 
                done[:, np.newaxis].astype(np.float32)
                )

    def clear(self):
        """
        clear the DataFrame.
        """
        self.data.drop(self.data.index, inplace=True)

    def get_max_episode(self):
        """
        get the max episode of this training model.
        """
        return self.max_episode
