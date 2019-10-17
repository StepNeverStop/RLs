import numpy as np
import tensorflow as tf
import Nn
from utils.sth import sth
from .policy import Policy


class MAXSQN(Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 action_type,
                 gamma=0.99,
                 max_episode=50000,
                 batch_size=128,
                 buffer_size=10000,
                 use_priority=False,
                 n_step=False,
                 base_dir=None,

                 alpha=0.2,
                 ployak=0.995,
                 lr=5.0e-4,
                 auto_adaption=True,
                 logger2file=False,
                 out_graph=False):
        assert action_type == 'discrete', 'maxsqn only support continuous action space'
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim_or_list=a_dim_or_list,
            action_type=action_type,
            gamma=gamma,
            max_episode=max_episode,
            base_dir=base_dir,
            policy_mode='OFF',
            batch_size=batch_size,
            buffer_size=buffer_size,
            use_priority=use_priority,
            n_step=n_step)
        self.lr = lr
        self.ployak = ployak
        self.log_alpha = alpha if not auto_adaption else tf.Variable(initial_value=0.0, name='log_alpha', dtype=tf.float64, trainable=True)
        self.auto_adaption = auto_adaption
        self.target_alpha = np.log(self.a_counts)
        self.q1_net = Nn.critic_q_all(self.s_dim, self.visual_dim, self.a_counts, 'q1_net')
        self.q1_target_net = Nn.critic_q_all(self.s_dim, self.visual_dim, self.a_counts, 'q1_target_net')
        self.q2_net = Nn.critic_q_all(self.s_dim, self.visual_dim, self.a_counts, 'q2_net')
        self.q2_target_net = Nn.critic_q_all(self.s_dim, self.visual_dim, self.a_counts, 'q2_target_net')
        self.update_target_net_weights(
            self.q1_target_net.weights + self.q2_target_net.weights,
            self.q1_net.weights + self.q2_net.weights
        )
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.optimizer_alpha = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.generate_recorder(
            logger2file=logger2file,
            model=self
        )
        self.recorder.logger.info('''
　　ｘｘｘｘ　　　　ｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘ　　　　　　　ｘｘｘｘ　　　ｘｘｘ　　　
　　　ｘｘｘ　　　　ｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘｘ　ｘｘｘ　　　　　　　　ｘｘｘ　　　ｘ　　　　
　　　　ｘｘｘ　　ｘｘｘ　　　　　　　　　ｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　　ｘ　　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘｘ　　　ｘ　　　　
　　　　ｘｘｘ　　ｘｘｘ　　　　　　　　ｘｘｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘ　　　　　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘ　ｘｘ　　ｘ　　　　
　　　　ｘｘｘｘ　ｘ　ｘ　　　　　　　ｘｘ　　ｘ　　　　　　　　　　　　ｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　ｘ　　　　　ｘ　　　　　　　　ｘ　ｘｘｘ　ｘ　　　　
　　　　ｘ　ｘｘｘｘ　ｘ　　　　　　　　ｘｘｘｘ　　　　　　　　　　　　ｘｘｘ　　　　　　　　　　　　　　ｘｘｘ　　　　　　　　ｘｘ　　　　ｘ　　　　　　　　ｘ　　ｘｘｘｘ　　　　
　　　　ｘ　ｘｘｘ　　ｘ　　　　　　　ｘｘｘ　ｘ　　　　　　　　　　　　ｘｘｘ　　　　　　　　　　ｘ　　　　ｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　
　　　　ｘ　　ｘｘ　　ｘ　　　　　　　ｘｘ　ｘｘ　　　　　　　　　　　　ｘｘｘｘ　　　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘｘ　ｘｘｘ　　　　　　　　ｘ　　　　ｘｘ　　　　
　　ｘｘｘｘ　ｘｘｘｘｘｘ　　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘｘｘ　ｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘ　ｘ　　　　　　ｘｘｘｘ　　　ｘ　　　　
　　　　　　　　　　　　　　　　　　　　　ｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘ　ｘｘｘ　　　　　　　　　　　　　　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　　　　　　　　　　　　　
        ''')

    def choose_action(self, s, visual_s):
        if np.random.uniform() < 0.2:
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
            q = self.q1_net(vector_input, visual_input)
            log_probs = tf.nn.log_softmax(q / tf.exp(self.log_alpha), axis=1)
        return tf.argmax(log_probs, axis=1)

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        self.off_store(s, visual_s, a, r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis])

    def learn(self, episode):
        self.episode = episode
        if self.data.is_lg_batch_size:
            s, visual_s, a, r, s_, visual_s_, done = self.data.sample()
            _a = sth.action_index2one_hot(a, self.a_dim_or_list)
            if self.use_priority:
                self.IS_w = self.data.get_IS_w()
            loss, entropy, td_error = self.train(s, visual_s, _a, r, s_, visual_s_, done)
            if self.use_priority:
                self.data.update(td_error, episode)
            self.update_target_net_weights(
                self.q1_target_net.weights + self.q2_target_net.weights,
                self.q1_net.weights + self.q2_net.weights,
                self.ployak)
            tf.summary.experimental.set_step(self.global_step)
            tf.summary.scalar('LOSS/loss', loss)
            tf.summary.scalar('LOSS/alpha', tf.exp(self.log_alpha))
            tf.summary.scalar('LOSS/entropy', entropy)
            tf.summary.scalar('LEARNING_RATE/lr', self.lr)
            self.recorder.writer.flush()

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, r, s_, visual_s_, done):
        done = tf.cast(done, tf.float64)
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                q1 = self.q1_net(s, visual_s)
                q1_eval = tf.reduce_sum(tf.multiply(q1, a), axis=1, keepdims=True)
                q2 = self.q2_net(s, visual_s)
                q2_eval = tf.reduce_sum(tf.multiply(q2, a), axis=1, keepdims=True)
                q1_target = self.q1_target_net(s_, visual_s_)
                q1_target_log_probs = tf.nn.log_softmax(q1_target / tf.exp(self.log_alpha), axis=1)
                q1_target_max = tf.reduce_max(q1_target, axis=1, keepdims=True)
                # q1_target_entropy = tf.reduce_sum(tf.exp(q1_target_log_probs) * q1_target_log_probs, axis=1, keepdims=True)
                q1_target_log_max = tf.reduce_max(q1_target_log_probs, axis=1, keepdims=True)

                q2_target = self.q2_target_net(s_, visual_s_)
                q2_target_max = tf.reduce_max(q2_target, axis=1, keepdims=True)

                q_target = tf.minimum(q1_target_max, q2_target_max) - tf.exp(self.log_alpha) * q1_target_log_max
                dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                td_error1 = q1_eval - dc_r
                td_error2 = q2_eval - dc_r
                q1_loss = tf.reduce_mean(tf.square(td_error1) * self.IS_w)
                q2_loss = tf.reduce_mean(tf.square(td_error2) * self.IS_w)
                loss = 0.5 * (q1_loss + q2_loss)
            loss_grads = tape.gradient(loss, self.q1_net.trainable_variables + self.q2_net.trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(loss_grads, self.q1_net.trainable_variables + self.q2_net.trainable_variables)
            )
            if self.auto_adaption:
                with tf.GradientTape() as tape:
                    q1 = self.q1_net(s, visual_s)
                    q1_log_probs = tf.nn.log_softmax(q1_target / tf.exp(self.log_alpha), axis=1)
                    q1_log_max = tf.reduce_max(q1_log_probs, axis=1, keepdims=True)
                    q1_entropy = tf.reduce_sum(tf.exp(q1_log_probs) * q1_log_probs, axis=1, keepdims=True)
                    alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(q1_log_max - self.target_alpha))
                alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
                self.optimizer_alpha.apply_gradients(
                    zip(alpha_grads, [self.log_alpha])
                )
            self.global_step.assign_add(1)
            return loss, tf.reduce_mean(q1_entropy), td_error1
