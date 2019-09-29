import numpy as np
import tensorflow as tf
import Nn
from .policy import Policy


class DDPG(Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 action_type,
                 gamma=0.99,
                 ployak=0.995,
                 lr=5.0e-4,
                 max_episode=50000,
                 batch_size=100,
                 buffer_size=10000,
                 base_dir=None,
                 logger2file=False,
                 out_graph=False):
        assert action_type == 'continuous', 'ddpg only support continuous action space'
        super().__init__(s_dim, visual_sources, visual_resolution, a_dim_or_list, action_type, gamma, max_episode, base_dir, 'OFF', batch_size, buffer_size)
        self.ployak = ployak
        self.lr = lr
        # self.action_noise = Nn.NormalActionNoise(mu=np.zeros(self.a_counts), sigma=1 * np.ones(self.a_counts))
        self.action_noise = Nn.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_counts), sigma=0.2 * np.ones(self.a_counts))
        self.actor_net = Nn.actor_dpg(self.s_dim, self.visual_dim, self.a_counts, 'actor')
        self.actor_target_net = Nn.actor_dpg(self.s_dim, self.visual_dim, self.a_counts, 'actor_target')
        self.q_net = Nn.critic_q_one(self.s_dim, self.visual_dim, self.a_counts, 'q')
        self.q_target_net = Nn.critic_q_one(self.s_dim, self.visual_dim, self.a_counts, 'q_target')
        self.update_target_net_weights(
            self.actor_target_net.weights + self.q_target_net.weights,
            self.actor_net.weights + self.q_net.weights,
            self.ployak)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.generate_recorder(
            logger2file=logger2file,
            model=self
        )
        self.recorder.logger.info('''
　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　　　　　　　
　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘ　　　ｘｘｘｘｘ　　　
　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　ｘｘｘ　　　　
　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　
        ''')
        self.recorder.logger.info(self.action_noise)

    def choose_action(self, s, visual_s):
        return self._get_action(s, visual_s)[-1].numpy()

    def choose_inference_action(self, s, visual_s):
        return self._get_action(s, visual_s)[0].numpy()

    @tf.function
    def _get_action(self, vector_input, visual_input):
        with tf.device(self.device):
            mu = self.actor_net(vector_input, visual_input)
        return mu, tf.clip_by_value(mu + self.action_noise(), -1, 1)

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        self.off_store(s, visual_s, a, r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis])

    def learn(self, episode):
        if self.data.is_lg_batch_size:
            s, visual_s, a, r, s_, visual_s_, done = self.data.sample()
            self.global_step.assign_add(1)
            actor_loss, q_loss = self.train(s, visual_s, a, r, s_, visual_s_, done)
            self.update_target_net_weights(
                self.actor_target_net.weights + self.q_target_net.weights,
                self.actor_net.weights + self.q_net.weights,
                self.ployak)
            tf.summary.experimental.set_step(self.global_step)
            tf.summary.scalar('LOSS/actor_loss', tf.reduce_mean(actor_loss))
            tf.summary.scalar('LOSS/critic_loss', tf.reduce_mean(q_loss))
            tf.summary.scalar('LEARNING_RATE/lr', tf.reduce_mean(self.lr))
            self.recorder.writer.flush()

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, r, s_, visual_s_, done):
        done = tf.cast(done, tf.float64)
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                target_mu = self.actor_target_net(s_, visual_s_)
                action_target = tf.clip_by_value(target_mu + self.action_noise(), -1, 1)
                q = self.q_net(s, visual_s, a)
                q_target = self.q_target_net(s_, visual_s_, action_target)
                dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                td_error = q - dc_r
                q_loss = 0.5 * tf.reduce_mean(tf.square(td_error))
            q_grads = tape.gradient(q_loss, self.q_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(q_grads, self.q_net.trainable_variables)
            )
            with tf.GradientTape() as tape:
                mu = self.actor_net(s, visual_s)
                q_actor = self.q_net(s, visual_s, mu)
                actor_loss = -tf.reduce_mean(q_actor)
            actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(actor_grads, self.actor_net.trainable_variables)
            )
            return actor_loss, q_loss
