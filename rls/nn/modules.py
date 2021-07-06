

import tensorflow as tf

from tensorflow.keras import Model as M
from tensorflow.keras import Input as I
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

from rls.utils.build_networks import DefaultRepresentationNetwork
from rls.utils.tf2_utils import get_device
from rls.nn.activations import default_activation
from rls.nn.initializers import initKernelAndBias


class CuriosityModel(M):
    '''
    Model of Intrinsic Curiosity Module (ICM).
    Curiosity-driven Exploration by Self-supervised Prediction, https://arxiv.org/abs/1705.05363
    '''

    def __init__(self,
                 obs_spec,
                 representation_net_params,
                 is_continuous,
                 action_dim,
                 *,
                 eta=0.2, lr=1.0e-3, beta=0.2):
        '''
        params:
            is_continuous: sepecify whether action space is continuous(True) or discrete(False)
            action_dim: dimension of action

            eta: weight of intrinsic reward
            lr: the learning rate of curiosity model
            beta: weight factor of loss between inverse_dynamic_net and forward_net
        '''
        super().__init__()
        self.device = get_device()
        self.eta = eta
        self.beta = beta
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.is_continuous = is_continuous

        self.repre_net = DefaultRepresentationNetwork(
            name='curiosity_model',
            obs_spec=obs_spec,
            representation_net_params=representation_net_params
        )

        self.feat_dim = self.repre_net.h_dim

        # S, S' => A
        self.inverse_dynamic_net = Sequential([
            Dense(self.feat_dim * 2, default_activation, **initKernelAndBias),
            Dense(action_dim, 'tanh' if is_continuous else None, **initKernelAndBias)
        ])

        # S, A => S'
        self.forward_net = Sequential([
            Dense(self.feat_dim + action_dim, default_activation, **initKernelAndBias),
            Dense(self.feat_dim, None, **initKernelAndBias)
        ])
        self.initial_weights(I(shape=self.feat_dim), I(shape=action_dim))

    def initial_weights(self, feat, action):
        self.inverse_dynamic_net(tf.concat((feat, feat), -1))
        self.forward_net(tf.concat((feat, action), -1))

    @tf.function
    def call(self, BATCH, cell_states):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                fs, _ = self.repre_net(BATCH.obs, cell_state=cell_states['obs'])
                fs_, _ = self.repre_net(BATCH.obs_, cell_state=cell_states['obs_'])

                fsa = tf.concat((fs, BATCH.action), axis=-1)            # <S, A>
                s_eval = self.forward_net(fsa)                  # <S, A> => S'
                LF = 0.5 * tf.reduce_sum(tf.square(fs_ - s_eval), axis=-1, keepdims=True)    # [B, 1]
                intrinsic_reward = self.eta * LF
                loss_forward = tf.reduce_mean(LF)

                f = tf.concat((fs, fs_), axis=-1)
                a_eval = self.inverse_dynamic_net(f)
                if self.is_continuous:
                    loss_inverse = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(a_eval - BATCH.action), axis=-1))
                else:
                    idx = tf.argmax(BATCH.action, axis=-1)  # [B, ]
                    loss_inverse = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=idx, logits=a_eval))
                loss = (1 - self.beta) * loss_inverse + self.beta * loss_forward

            grads = tape.gradient(loss, self.repre_net.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.repre_net.trainable_variables))
            summaries = dict([
                ['LOSS/curiosity_loss', loss],
                ['LOSS/forward_loss', loss_forward],
                ['LOSS/inverse_loss', loss_inverse]
            ])
            return intrinsic_reward, summaries
