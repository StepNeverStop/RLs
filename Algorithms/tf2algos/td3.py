import numpy as np
import tensorflow as tf
import Nn
from .policy import Policy


class TD3(Policy):
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
        assert action_type == 'continuous', 'td3 only support continuous action space'
        super().__init__(s_dim, visual_sources, visual_resolution, a_dim_or_list, action_type, gamma, max_episode, base_dir, 'OFF', batch_size, buffer_size)
        self.ployak = ployak
        self.lr = lr
        # self.action_noise = Nn.NormalActionNoise(mu=np.zeros(self.a_counts), sigma=1 * np.ones(self.a_counts))
        self.action_noise = Nn.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_counts), sigma=0.2 * np.ones(self.a_counts))
        self.actor_net = Nn.actor_dpg(self.s_dim, self.visual_dim, self.a_counts, 'actor')
        self.actor_target_net = Nn.actor_dpg(self.s_dim, self.visual_dim, self.a_counts, 'actor_target')
        self.q1_net = Nn.critic_q_one(self.s_dim, self.visual_dim, self.a_counts, 'q1')
        self.q1_target_net = Nn.critic_q_one(self.s_dim, self.visual_dim, self.a_counts, 'q1_target')
        self.q2_net = Nn.critic_q_one(self.s_dim, self.visual_dim, self.a_counts, 'q2')
        self.q2_target_net = Nn.critic_q_one(self.s_dim, self.visual_dim, self.a_counts, 'q2_target')
        self.update_target_net_weights(
            self.actor_target_net.weights + self.q1_target_net.weights + self.q2_target_net.weights,
            self.actor_net.weights + self.q1_net.weights + self.q2_net.weights,
            self.ployak)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.generate_recorder(
            logger2file=logger2file,
            model=self
        )
        self.recorder.logger.info('''
　　　ｘｘｘｘｘｘｘｘｘ　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　　　　　
　　　ｘｘ　　ｘ　　ｘｘ　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　ｘｘ　　　　　
　　　ｘｘ　　ｘ　　ｘｘ　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘｘ　ｘｘ　　　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　　　ｘｘｘ　　　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　　ｘｘｘｘ　　　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　　　　ｘｘｘ　　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　ｘｘｘ　　　　
　　　　　ｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　
        ''')

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
            actor_loss, critic_loss = self.train(s, visual_s, a, r, s_, visual_s_, done)
            self.update_target_net_weights(
                self.actor_target_net.weights + self.q1_target_net.weights + self.q2_target_net.weights,
                self.actor_net.weights + self.q1_net.weights + self.q2_net.weights,
                self.ployak)
            tf.summary.experimental.set_step(self.global_step)
            tf.summary.scalar('LOSS/actor_loss', tf.reduce_mean(actor_loss))
            tf.summary.scalar('LOSS/critic_loss', tf.reduce_mean(critic_loss))
            tf.summary.scalar('LEARNING_RATE/lr', tf.reduce_mean(self.lr))
            self.recorder.writer.flush()

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, r, s_, visual_s_, done):
        done = tf.cast(done, tf.float64)
        with tf.device(self.device):
            for _ in range(2):
                with tf.GradientTape() as tape:
                    target_mu = self.actor_net(s_, visual_s_)
                    action_target = tf.clip_by_value(target_mu + self.action_noise(), -1, 1)
                    q1 = self.q1_net(s, visual_s, a)
                    q1_target = self.q1_target_net(s_, visual_s_, action_target)
                    q2 = self.q2_net(s, visual_s, a)
                    q2_target = self.q2_target_net(s_, visual_s_, action_target)
                    q_target = tf.minimum(q1_target, q2_target)
                    dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                    td_error1 = q1 - dc_r
                    td_error2 = q2 - dc_r
                    q1_loss = tf.reduce_mean(tf.square(td_error1))
                    q2_loss = tf.reduce_mean(tf.square(td_error2))
                    critic_loss = 0.5 * (q1_loss + q2_loss)
                critic_grads = tape.gradient(critic_loss, self.q1_net.trainable_variables + self.q2_net.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(critic_grads, self.q1_net.trainable_variables + self.q2_net.trainable_variables)
                )
            with tf.GradientTape() as tape:
                mu = self.actor_net(s, visual_s)
                q1_actor = self.q1_net(s, visual_s, mu)
                actor_loss = -tf.reduce_mean(q1_actor)
            actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(actor_grads, self.actor_net.trainable_variables)
            )
            return actor_loss, critic_loss
