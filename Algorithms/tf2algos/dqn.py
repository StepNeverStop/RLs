import Nn
import numpy as np
import tensorflow as tf
from utils.sth import sth
from .policy import Policy


class DQN(Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 action_type,
                 lr=5.0e-4,
                 gamma=0.99,
                 epsilon=0.2,
                 max_episode=50000,
                 batch_size=100,
                 buffer_size=10000,
                 assign_interval=2,
                 use_priority=False,
                 n_step=False,
                 base_dir=None,
                 logger2file=False,
                 out_graph=False):
        assert action_type == 'discrete', 'dqn only support discrete action space'
        super().__init__(s_dim, visual_sources, visual_resolution, a_dim_or_list, action_type, gamma, max_episode,
                         base_dir, 'OFF', batch_size=batch_size, buffer_size=buffer_size, use_priority=use_priority, n_step=n_step)
        self.epsilon = epsilon
        self.assign_interval = assign_interval
        self.lr = lr
        self.q_net = Nn.critic_q_all(self.s_dim, self.visual_dim, self.a_counts, 'q')
        self.q_target_net = Nn.critic_q_all(self.s_dim, self.visual_dim, self.a_counts, 'q_target')
        self.update_target_net_weights(self.q_target_net.weights, self.q_net.weights)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.generate_recorder(
            logger2file=logger2file,
            model=self
        )
        self.recorder.logger.info('''
    　　　ｘｘｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　ｘｘｘｘ　　　ｘｘｘｘ　　
    　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘ　ｘｘｘｘ　　　　　　　ｘｘｘ　　　　ｘ　　　
    　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　ｘｘｘｘ　　　　　　ｘｘｘｘ　　　ｘ　　　
    　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘｘｘｘｘ　　ｘ　　　
    　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘｘ　　　　　　ｘ　ｘｘｘｘ　ｘ　　　
    　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘ　　ｘｘｘｘｘ　　　
    　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘ　　　ｘｘｘｘ　　　
    　　　　ｘｘ　　　ｘｘｘｘ　　　　　ｘｘｘ　　　ｘｘｘ　　　　　　　ｘ　　　　ｘｘｘ　　　
    　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　ｘｘｘ　　　　ｘｘ　　　
    　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　　　　　　　　　　
    　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘ　　　　　　　　　　　　　　　　　　　
    　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘ
        ''')

    def choose_action(self, s, visual_s):
        if np.random.uniform() < self.epsilon:
            a = np.random.randint(0, self.a_counts, len(s))
        else:
            a = self._get_action(s, visual_s).numpy()
        return sth.int2action_index(a, self.a_dim_or_list)

    def choose_inference_action(self, s, visual_s):
        return sth.int2action_index(
            self._get_action(s, visual_s).numpy(),
            self.a_dim_or_list
        )

    @tf.function
    def _get_action(self, vector_input, visual_input):
        with tf.device(self.device):
            q_values = self.q_net(vector_input, visual_input)
        return tf.argmax(q_values, axis=1)

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        self.data.add(s, visual_s, a, r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis])

    def learn(self, episode):
        self.episode = episode
        if self.data.is_lg_batch_size:
            s, visual_s, a, r, s_, visual_s_, done = self.data.sample()
            _a = sth.action_index2one_hot(a, self.a_dim_or_list)
            self.global_step.assign_add(1)
            q_loss = self.train(s, visual_s, _a, r, s_, visual_s_, done)
            if self.global_step % self.assign_interval == 0:
                self.update_target_net_weights(self.q_target_net.weights, self.q_net.weights)
            tf.summary.experimental.set_step(self.global_step)
            tf.summary.scalar('LOSS/loss', tf.reduce_mean(q_loss))
            tf.summary.scalar('LEARNING_RATE/lr', tf.reduce_mean(self.lr))
            self.recorder.writer.flush()

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, r, s_, visual_s_, done):
        done = tf.cast(done, tf.float64)
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                q = self.q_net(s, visual_s)
                q_next = self.q_target_net(s_, visual_s_)
                q_eval = tf.reduce_sum(tf.multiply(q, a), axis=1, keepdims=True)
                q_target = tf.stop_gradient(r + self.gamma * (1 - done) * tf.reduce_max(q_next, axis=1, keepdims=True))
                td_error = q_eval - q_target
                q_loss = tf.reduce_mean(tf.square(td_error))
            grads = tape.gradient(q_loss, self.q_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.q_net.trainable_variables)
            )
            return q_loss
