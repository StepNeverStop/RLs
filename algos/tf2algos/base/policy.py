import numpy as np
import tensorflow as tf
from .base import Base
from abc import abstractmethod
from rls.networks import CuriosityModel
from rls.learningrate import ConsistentLearningRate
from utils.vector_runing_average import DefaultRunningAverage, SimpleRunningAverage


class Policy(Base):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim,
                 is_continuous,
                 **kwargs):
        super().__init__(**kwargs)
        self.s_dim = s_dim
        self.feat_dim = self.s_dim
        self.visual_sources = visual_sources
        if visual_sources >= 1:
            self.use_visual = True
            self.visual_dim = [visual_sources, *visual_resolution]
        else:
            self.use_visual = False
            self.visual_dim = [0]

        self.use_rnn = bool(kwargs.get('use_rnn', False))

        self._normalize_vector_obs = bool(kwargs.get('normalize_vector_obs', False))
        self._running_average = SimpleRunningAverage(dim=self.s_dim) if self._normalize_vector_obs else DefaultRunningAverage()

        self.other_tv = []

        self.batch_size = int(kwargs.get('batch_size', 128))
        self.n_agents = int(kwargs.get('n_agents', 0))
        if self.n_agents <= 0:
            raise ValueError('agents num must larger than zero.')

        self.is_continuous = is_continuous
        self.a_dim = a_dim
        self.gamma = float(kwargs.get('gamma', 0.999))
        self.train_step = 0
        self.max_train_step = int(kwargs.get('max_train_step', 1000))
        self.delay_lr = bool(kwargs.get('decay_lr', True))

        self.use_curiosity = bool(kwargs.get('use_curiosity', False))
        if self.use_curiosity:
            self.curiosity_eta = float(kwargs.get('curiosity_reward_eta'))
            self.curiosity_lr = float(kwargs.get('curiosity_lr'))
            self.curiosity_beta = float(kwargs.get('curiosity_beta'))
            self.curiosity_loss_weight = float(kwargs.get('curiosity_loss_weight'))
            self.curiosity_model = CuriosityModel(self.is_continuous, self.s_dim, self.a_dim, self.visual_dim, 128,
                                                  eta=self.curiosity_eta, lr=self.curiosity_lr, beta=self.curiosity_beta, loss_weight=self.curiosity_loss_weight)
        self.writer = self._create_writer(self.log_dir) # TODO: Annotation

    def init_lr(self, lr):
        if self.delay_lr:
            return tf.keras.optimizers.schedules.PolynomialDecay(lr, self.max_train_step, 1e-10, power=1.0)
        else:
            return ConsistentLearningRate(lr)

    def normalize_vector_obs(self, x):
        return self._running_average.normalize(x)

    def init_optimizer(self, lr, *args, **kwargs):
        return tf.keras.optimizers.Adam(learning_rate=lr(self.train_step), *args, **kwargs)

    def reset(self):
        self.cell_state = (None,)

    def get_cell_state(self):
        return self.cell_state

    def set_cell_state(self, cs):
        pass

    def partial_reset(self, done):
        pass

    def model_recorder(self, kwargs):
        kwargs.update(dict(global_step=self.global_step))
        if self.use_curiosity:
            kwargs.update(curiosity_model=self.curiosity_model)
        self._create_saver(kwargs)
        self.show_logo()

    def intermediate_variable_reset(self):
        '''
        TODO: Annotation
        '''
        self.summaries = {}

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
            tf.group([t.assign(ployak * t + (1 - ployak) * s) for t, s in zip(tge, src)])

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
