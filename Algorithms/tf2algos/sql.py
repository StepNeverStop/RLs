import Nn
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from Algorithms.tf2algos.base.off_policy import Off_Policy


class SQL(Off_Policy):
    '''
        Soft Q-Learning.
        Reinforcement Learning with Deep Energy-Based Policies: https://arxiv.org/abs/1702.08165
    '''
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,

                 lr=5.0e-4,
                 alpha=2,
                 ployak=0.995,
                 hidden_units=[32, 32],
                 **kwargs):
        assert not is_continuous, 'sql only support discrete action space'
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim_or_list=a_dim_or_list,
            is_continuous=is_continuous,
            **kwargs)
        self.alpha = alpha
        self.ployak = ployak

        _q_net = lambda : Nn.critic_q_all(self.rnn_net.hdim, self.a_counts, hidden_units)

        self.q_net = _q_net()
        self.q_target_net = _q_net()
        self.critic_tv = self.q_net.trainable_variables + self.other_tv
        self.lr = tf.keras.optimizers.schedules.PolynomialDecay(lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr(self.episode))

        self.update_target_net_weights(
            self.q_target_net.weights,
            self.q_net.weights
        )

        self.model_recorder(dict(
            model=self.q_net,
            optimizer=self.optimizer
        ))

    def show_logo(self):
        self.recorder.logger.info('''
　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　
　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘｘｘｘ　　　　　　　　　ｘｘｘ　　　　　　　　　　
　　　　　ｘｘｘ　　ｘｘｘ　　　　　　　　　ｘｘｘｘ　　　ｘｘｘ　　　　　　　　　　ｘｘ　　　　　　　　　　
　　　　　ｘｘｘ　　　ｘｘ　　　　　　　　　ｘｘｘ　　　　ｘｘｘｘ　　　　　　　　　ｘｘ　　　　　　　　　　
　　　　　ｘｘｘｘｘ　　　　　　　　　　　ｘｘｘ　　　　　　ｘｘｘ　　　　　　　　　ｘｘ　　　　　　　　　　
　　　　　　ｘｘｘｘｘ　　　　　　　　　　ｘｘｘ　　　　　　ｘｘｘ　　　　　　　　　ｘｘ　　　　　　　　　　
　　　　　　　ｘｘｘｘｘｘ　　　　　　　　ｘｘｘ　　　　　　ｘｘｘ　　　　　　　　　ｘｘ　　　　　　　　　　
　　　　　　　　　ｘｘｘｘ　　　　　　　　ｘｘｘ　　　　　　ｘｘｘ　　　　　　　　　ｘｘ　　　　　　　　　　
　　　　　ｘｘ　　　ｘｘｘｘ　　　　　　　　ｘｘｘ　　　　　ｘｘｘ　　　　　　　　　ｘｘ　　　　　ｘ　　　　
　　　　　ｘｘ　　　　ｘｘ　　　　　　　　　ｘｘｘｘ　　　ｘｘｘ　　　　　　　　　　ｘｘ　　　　ｘｘ　　　　
　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘｘｘ　　　　　
　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘ　　
        ''')

    def choose_action(self, s, visual_s, evaluation=False):
        a, self.cell_state = self._get_action(s, visual_s, self.cell_state)
        a = a.numpy()
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state):
        with tf.device(self.device):
            feat, cell_state = self.get_feature(s, visual_s, cell_state=cell_state, record_cs=True, train=False)
            q_values = self.q_net(feat)
            logits = tf.math.exp((q_values-self.get_v(q_values))/self.alpha)
            cate_dist = tfp.distributions.Categorical(logits)
            pi = cate_dist.sample()
        return pi, cell_state

    @tf.function
    def get_v(self, q):
        with tf.device(self.device):
            v = self.alpha*tf.math.log(tf.reduce_mean(tf.math.exp(q/self.alpha), axis=1, keepdims=True))
        return v

    def learn(self, **kwargs):
        self.episode = kwargs['episode']
        for i in range(kwargs['step']):
            self._learn(function_dict={
                'train_function': self.train,
                'update_function': lambda : self.update_target_net_weights(
                                            self.q_target_net.weights,
                                            self.q_net.weights,
                                            self.ployak),
                'summary_dict': dict([['LEARNING_RATE/lr', self.lr(self.episode)]])
            })

    @tf.function(experimental_relax_shapes=True)
    def train(self, memories, isw, crsty_loss, cell_state):
        ss, vvss, a, r, done = memories
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                q = self.q_net(feat)
                q_next = self.q_target_net(feat_)
                v_next = self.get_v(q_next)
                q_eval = tf.reduce_sum(tf.multiply(q, a), axis=1, keepdims=True)
                q_target = tf.stop_gradient(r + self.gamma * (1 - done) * v_next)
                td_error = q_eval - q_target
                q_loss = tf.reduce_mean(tf.square(td_error) * isw) + crsty_loss
            grads = tape.gradient(q_loss, self.critic_tv)
            self.optimizer.apply_gradients(
                zip(grads, self.critic_tv)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/loss', q_loss],
                ['Statistics/q_max', tf.reduce_max(q_eval)],
                ['Statistics/q_min', tf.reduce_min(q_eval)],
                ['Statistics/q_mean', tf.reduce_mean(q_eval)]
            ])
