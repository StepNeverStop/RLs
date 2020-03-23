import numpy as np
import tensorflow as tf
from .base import Base
from abc import abstractmethod
from Nn.networks import CuriosityModel, VisualNet, ObsRNN


class Policy(Base):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,
                 **kwargs):
        super().__init__(**kwargs)
        self.s_dim = s_dim
        self.visual_sources = visual_sources
        if visual_sources == 1:
            self.visual_dim = visual_resolution
        elif visual_sources > 1:
            self.visual_dim = [visual_sources, *visual_resolution]
        else:
            self.visual_dim = [0]
        self.visual_feature = int(kwargs.get('visual_feature', 128))
        self.visual_net = VisualNet(self.s_dim, self.visual_dim, self.visual_feature)

        self.batch_size = int(kwargs.get('batch_size', 128))

        self.use_rnn = bool(kwargs.get('use_rnn', False))
        self.rnn_units = int(kwargs.get('rnn_units', 16))
        self.rnn_net = ObsRNN(self.visual_net.hdim, self.rnn_units, self.batch_size, self.use_rnn)
        self.cell_state = None

        self.other_tv = self.visual_net.trainable_variables + self.rnn_net.trainable_variables

        self.is_continuous = is_continuous
        self.a_dim_or_list = a_dim_or_list
        self.gamma = float(kwargs.get('gamma', 0.999))
        self.max_episode = int(kwargs.get('max_episode', 1000))
        self.a_counts = int(np.asarray(a_dim_or_list).prod())
        self.episode = 0    # episode of now
        self.IS_w = 1       # the weights of NN variables by using Importance sampling.
        self.curiosity_loss_constant = 0.

        self.use_curiosity = bool(kwargs.get('use_curiosity', False))
        if self.use_curiosity:
            self.curiosity_eta = float(kwargs.get('curiosity_reward_eta'))
            self.curiosity_lr = float(kwargs.get('curiosity_lr'))
            self.curiosity_beta = float(kwargs.get('curiosity_beta'))
            self.curiosity_loss_weight = float(kwargs.get('curiosity_loss_weight'))
            self.curiosity_model = CuriosityModel(self.is_continuous, self.s_dim, self.a_counts, self.visual_dim, 128, 
                                                  eta=self.curiosity_eta, lr=self.curiosity_lr, beta=self.curiosity_beta, loss_weight=self.curiosity_loss_weight)

    @tf.function
    def get_feature(self, s, visual_s, use_cs=False, record_cs=False, train=True):
        with tf.device(self.device):
            feature = self.visual_net(s, visual_s)
            if use_cs:
                state, cell_state = self.rnn_net(feature, self.cell_state, train=train)
            else:
                state, cell_state = self.rnn_net(feature, train=train)
            if record_cs:
                self.cell_state = cell_state
            return state

    def set_buffer(self, buffer):
        pass

    def model_recorder(self, kwargs):
        kwargs.update(dict(
            global_step=self.global_step,
            visual_net=self.visual_net,
            rnn_net=self.rnn_net))
        if self.use_curiosity:
            kwargs.update(curiosity_model=self.curiosity_model)
        self.generate_recorder(kwargs)
        self.show_logo()

    def intermediate_variable_reset(self):
        '''
        TODO: Annotation
        '''
        self.summaries = {}
        self.curiosity_loss_constant = 0.

    def get_max_episode(self):
        """
        get the max episode of this training model.
        """
        return self.max_episode

    @abstractmethod
    def choose_action(self, s, visual_s, evaluation=False):
        '''
        choose actions while training.
        Input: 
            s: vector observation
            visual_s: visual observation
        Output: 
            actions
        '''
        pass

    def update_target_net_weights(self, tge, src, ployak=None):
        '''
        update weights of target neural network.
        '''
        if ployak is None:
            tf.group([t.assign(s) for t, s in zip(tge, src)])
        else:
            tf.group([t.assign(self.ployak * t + (1 - self.ployak) * s) for t, s in zip(tge, src)])

    @tf.function
    def _get_action(self, s, visual_s, is_training=True):
        '''
        TODO: Annotation
        '''
        raise NotImplementedError

    def set_buffer(self, buffer):
        '''
        TODO: Annotation
        '''
        pass
