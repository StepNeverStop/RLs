import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import Nn
from .policy import Policy


class SAC_NO_V(Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 action_type,
                 alpha=0.2,
                 auto_adaption=True,
                 lr=5.0e-4,
                 max_episode=50000,
                 gamma=0.99,
                 ployak=0.995,
                 batch_size=100,
                 buffer_size=10000,
                 base_dir=None,
                 logger2file=False,
                 out_graph=False):
        assert action_type == 'continuous', 'sac only support continuous action space'
        super().__init__(s_dim, visual_sources, visual_resolution, a_dim_or_list, action_type, gamma, max_episode, base_dir, 'OFF', batch_size, buffer_size)
        self.lr = lr
        self.ployak = ployak
        self.sigma_offset = np.full([self.a_counts, ], 0.01)
        self.log_alpha = alpha if not auto_adaption else tf.Variable(initial_value=0.0, name='log_alpha', dtype=tf.float64, trainable=True)
        self.auto_adaption = auto_adaption
        self.actor_net = Nn.actor_continuous(self.s_dim, self.visual_dim, self.a_counts, 'actor')
        self.q1_net = Nn.critic_q_one(self.s_dim, self.visual_dim, self.a_counts, 'q1')
        self.q1_target_net = Nn.critic_q_one(self.s_dim, self.visual_dim, self.a_counts, 'q1_target')
        self.q2_net = Nn.critic_q_one(self.s_dim, self.visual_dim, self.a_counts, 'q2')
        self.q2_target_net = Nn.critic_q_one(self.s_dim, self.visual_dim, self.a_counts, 'q2_target')
        self.update_target_net_weights(
            self.q1_target_net.weights + self.q2_target_net.weights,
            self.q1_net.weights + self.q2_net.weights,
            self.ployak)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.generate_recorder(
            logger2file=logger2file,
            model=self
        )
        self.recorder.logger.info('''
　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　　ｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘ　　ｘｘｘ　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　　　ｘｘｘ　　　　　　　　　　ｘｘｘ　　ｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘ　　　　ｘ　　　　
　　　　ｘｘ　　　　ｘ　　　　　　　　　　ｘｘｘ　　　　　　　　　　ｘｘ　　　　ｘｘ　　　　　　　ｘｘ　　　　　　　　　　　　　　　　ｘｘ　　　　　　　　　　ｘｘ　　　ｘｘ　　　　
　　　　ｘｘｘｘ　　　　　　　　　　　　　ｘ　ｘｘ　　　　　　　　　ｘｘ　　　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　ｘｘｘ　　ｘｘ　　　　
　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘｘ　ｘｘ　　　　　　　　ｘｘｘ　　　　　　　　　　　　　　ｘｘ　ｘｘ　　　　　　　　　　ｘｘ　　ｘｘ　　　　　　　　　ｘｘ　ｘｘ　　　　　
　　　　　　　　ｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　　　　　　　　　　　　　ｘ　　　ｘ　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘｘｘｘｘ　　　　　
　　　　ｘ　　　　ｘｘ　　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘ　　　　ｘｘ　　　　　　　　ｘ　　　ｘ　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　　ｘｘｘ　　　　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘｘ　　ｘｘｘ　　　　　　　　ｘ　　　ｘ　　　　　　　　　　ｘｘ　　ｘｘ　　　　　　　　　　ｘｘｘ　　　　　　
　　　　ｘｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　ｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘ　　　　　　　　ｘｘｘ　ｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　　ｘ　　　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　　　　　　　　　　　ｘ　　　
        ''')

    def choose_action(self, s, visual_s):
        return self._get_action(s, visual_s)[-1].numpy()

    def choose_inference_action(self, s, visual_s):
        return self._get_action(s, visual_s)[0].numpy()

    @tf.function
    def _get_action(self, vector_input, visual_input):
        with tf.device(self.device):
            mu, sigma = self.actor_net(vector_input, visual_input)
            norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma + self.sigma_offset)
            a_new = tf.clip_by_value(norm_dist.sample(), -1, 1)
        return mu, a_new

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        self.off_store(s, visual_s, a, r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis])

    def learn(self, episode):
        if self.data.is_lg_batch_size:
            s, visual_s, a, r, s_, visual_s_, done = self.data.sample()
            self.global_step.assign_add(1)
            actor_loss, critic_loss, entropy = self.train(s, visual_s, a, r, s_, visual_s_, done)
            self.update_target_net_weights(
                self.q1_target_net.weights + self.q2_target_net.weights,
                self.q1_net.weights + self.q2_net.weights,
                self.ployak)
            tf.summary.experimental.set_step(self.global_step)
            tf.summary.scalar('LOSS/actor_loss', actor_loss)
            tf.summary.scalar('LOSS/critic_loss', critic_loss)
            tf.summary.scalar('LOSS/alpha', tf.exp(self.log_alpha))
            tf.summary.scalar('LOSS/entropy', entropy)
            tf.summary.scalar('LEARNING_RATE/lr', self.lr)
            self.recorder.writer.flush()

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, r, s_, visual_s_, done):
        done = tf.cast(done, tf.float64)
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                target_mu, target_sigma = self.actor_net(s_, visual_s_)
                target_norm_dist = tfp.distributions.Normal(loc=target_mu, scale=target_sigma + self.sigma_offset)
                a_s_ = tf.clip_by_value(target_norm_dist.sample(), -1, 1)
                a_s_log_prob_ = target_norm_dist.log_prob(a_s_)
                q1 = self.q1_net(s, visual_s, a)
                q1_target = self.q1_target_net(s_, visual_s_, a_s_)
                q2 = self.q2_net(s, visual_s, a)
                q2_target = self.q2_target_net(s_, visual_s_, a_s_)
                dc_r_q1 = tf.stop_gradient(r + self.gamma * (q1_target - tf.exp(self.log_alpha) * tf.reduce_mean(a_s_log_prob_) * (1 - done)))
                dc_r_q2 = tf.stop_gradient(r + self.gamma * (q2_target - tf.exp(self.log_alpha) * tf.reduce_mean(a_s_log_prob_) * (1 - done)))
                td_error1 = q1 - dc_r_q1
                td_error2 = q2 - dc_r_q2
                q1_loss = tf.reduce_mean(tf.square(td_error1))
                q2_loss = tf.reduce_mean(tf.square(td_error2))
                critic_loss = 0.5 * q1_loss + 0.5 * q2_loss
            critic_grads = tape.gradient(critic_loss, self.q1_net.trainable_variables + self.q2_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(critic_grads, self.q1_net.trainable_variables + self.q2_net.trainable_variables)
            )

            with tf.GradientTape() as tape:
                mu, sigma = self.actor_net(s, visual_s)
                norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma + self.sigma_offset)
                a_new = tf.clip_by_value(norm_dist.sample(), -1, 1)
                a_s_log_prob = norm_dist.log_prob(a_new)
                entropy = tf.reduce_mean(norm_dist.entropy())
                q1_s_a = self.q1_net(s, visual_s, a_new)
                q2_s_a = self.q2_net(s, visual_s, a_new)
                actor_loss = -tf.reduce_mean(tf.minimum(q1_s_a, q2_s_a) - tf.exp(self.log_alpha) * a_s_log_prob)
            actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(actor_grads, self.actor_net.trainable_variables)
            )

            if self.auto_adaption:
                with tf.GradientTape() as tape:
                    mu, sigma = self.actor_net(s, visual_s)
                    norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma + self.sigma_offset)
                    a_new = tf.clip_by_value(norm_dist.sample(), -1, 1)
                    a_s_log_prob = norm_dist.log_prob(a_new)
                    alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(a_s_log_prob - self.a_counts))
                alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
                self.optimizer.apply_gradients(
                    zip(alpha_grads, [self.log_alpha])
                )
            return actor_loss, critic_loss, entropy
