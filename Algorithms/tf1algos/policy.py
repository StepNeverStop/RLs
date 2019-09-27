import numpy as np
import pandas as pd
import tensorflow as tf
import Nn
from .base import Base
from utils.replay_buffer import ExperienceReplay, NStepExperienceReplay, PrioritizedExperienceReplay, NStepPrioritizedExperienceReplay


class Policy(Base):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 action_type,
                 gamma,
                 max_episode,
                 base_dir,
                 policy_mode=None,
                 batch_size=1,
                 buffer_size=1,
                 use_priority=False,
                 n_step=False):
        super().__init__(a_dim_or_list, action_type, base_dir)
        self.s_dim = s_dim
        self.visual_sources = visual_sources
        self.visual_dim = [visual_sources, *visual_resolution] if visual_sources else [0]
        self.a_dim_or_list = a_dim_or_list
        self.gamma = gamma
        self.max_episode = max_episode
        self.policy_mode = policy_mode
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        '''
        the biggest diffenernce between policy_modes(ON and OFF) is 'OFF' mode need raise the dimension
        of 'r' and 'done'.
        'ON' mode means program will call on_store function and use pandas dataframe to store data.
        'OFF' mode will call off_store function and use replay buffer to store data.
        '''
        if self.policy_mode == 'ON':
            self.data = pd.DataFrame(columns=['s', 'a', 'r', 's_', 'done'])
        elif self.policy_mode == 'OFF':
            if use_priority:
                if n_step:
                    print('N-Step PER')
                    self.data = NStepPrioritizedExperienceReplay(self.batch_size, self.buffer_size, max_episode=self.max_episode,
                                                                 gamma=self.gamma, alpha=0.6, beta=0.2, epsilon=0.01, agents_num=20, n=4)
                else:
                    print('PER')
                    self.data = PrioritizedExperienceReplay(self.batch_size, self.buffer_size, max_episode=self.max_episode, alpha=0.6, beta=0.2, epsilon=0.01)
            else:
                if n_step:
                    print('N-Step ER')
                    self.data = NStepExperienceReplay(self.batch_size, self.buffer_size, gamma=self.gamma, agents_num=20, n=4)
                else:
                    print('ER')
                    self.data = ExperienceReplay(self.batch_size, self.buffer_size)
        else:
            raise Exception('Please specific a mode of policy!')

        with self.graph.as_default():
            self.pl_s = tf.placeholder(tf.float32, [None, self.s_dim], 'vector_observation')
            self.pl_a = tf.placeholder(tf.float32, [None, self.a_counts], 'pl_action')
            self.pl_r = tf.placeholder(tf.float32, [None, 1], 'reward')
            self.pl_s_ = tf.placeholder(tf.float32, [None, self.s_dim], 'next_state')
            self.pl_done = tf.placeholder(tf.float32, [None, 1], 'done')
            self.pl_visual_s = tf.placeholder(tf.float32, [None] + self.visual_dim, 'visual_observation_')
            self.pl_visual_s_ = tf.placeholder(tf.float32, [None] + self.visual_dim, 'next_visual_observation')

    def on_store(self, s, visual_s, a, r, s_, visual_s_, done):
        """
        for on-policy training, use this function to store <s, a, r, s_, done> into DataFrame of Pandas.
        """
        assert isinstance(a, np.ndarray), "on_store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "on_store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "on_store need done type is np.ndarray"
        self.data = self.data.append({
            's': s,
            'visual_s': visual_s,
            'a': a,
            'r': r,
            's_': s_,
            'visual_s_': visual_s_,
            'done': done
        }, ignore_index=True)

    def off_store(self, s, visual_s, a, r, s_, visual_s_, done):
        """
        for off-policy training, use this function to store <s, a, r, s_, done> into ReplayBuffer.
        """
        assert isinstance(a, np.ndarray), "off_store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "off_store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "off_store need done type is np.ndarray"
        self.data.add(s, visual_s, a, r, s_, visual_s_, done)

    def no_op_store(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "no_op_store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "no_op_store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "no_op_store need done type is np.ndarray"
        if self.policy_mode == 'OFF':
            self.data.add(s, visual_s, a, r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis])

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
