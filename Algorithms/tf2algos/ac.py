import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import Nn
from utils.sth import sth
from .policy import Policy


class AC(Policy):
    # off-policy actor-critic
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

                 lr=5.0e-4,
                 logger2file=False,
                 out_graph=False):
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
        self.sigma_offset = np.full([self.a_counts, ], 0.01)
        if self.action_type == 'continuous':
            self.actor_net = Nn.actor_continuous(self.s_dim, self.visual_dim, self.a_counts, 'actor_net')
        else:
            self.actor_net = Nn.actor_discrete(self.s_dim, self.visual_dim, self.a_counts, 'actor_net')
        self.critic_net = Nn.critic_q_one(self.s_dim, self.visual_dim, self.a_counts, 'critic_net')
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.generate_recorder(
            logger2file=logger2file,
            model=self
        )
        self.recorder.logger.info('''
　　　　　　　ｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　
　　　　　　ｘｘｘ　　　　　　　　　　ｘｘｘ　　ｘｘ　　　　
　　　　　　ｘｘｘ　　　　　　　　　　ｘｘ　　　　ｘｘ　　　
　　　　　　ｘ　ｘｘ　　　　　　　　　ｘｘ　　　　　　　　　
　　　　　ｘｘ　ｘｘ　　　　　　　　ｘｘｘ　　　　　　　　　
　　　　　ｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　　　　　　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘ　　　　ｘｘ　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘｘ　　ｘｘｘ　　　
　　　ｘｘｘ　　ｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘ　　　　　　　　
        ''')

    def choose_action(self, s, visual_s):
        if self.action_type == 'continuous':
            return self._get_action(s, visual_s).numpy()
        else:
            if np.random.uniform() < self.epsilon:
                a = np.random.randint(0, self.a_counts, len(s))
            else:
                a = self._get_action(s, visual_s).numpy()
            return sth.int2action_index(a, self.a_dim_or_list)

    def choose_inference_action(self, s, visual_s):
        a = self._get_action(s, visual_s).numpy()
        return a if self.action_type == 'continuous' else sth.int2action_index(a, self.a_dim_or_list)

    @tf.function
    def _get_action(self, vector_input, visual_input):
        with tf.device(self.device):
            if self.action_type == 'continuous':
                mu, sigma = self.actor_net(vector_input, visual_input)
                norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma + self.sigma_offset)
                sample_op = tf.clip_by_value(norm_dist.sample(), -1, 1)
            else:
                action_probs = self.actor_net(vector_input, visual_input)
                sample_op = tf.argmax(action_probs, axis=1)
        return sample_op

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        if not self.action_type == 'continuous':
            a = sth.action_index2one_hot(a, self.a_dim_or_list)
        old_prob = self._get_prob(s, visual_s, a)
        assert isinstance(a, np.ndarray), "store_data need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store_data need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store_data need done type is np.ndarray"
        self.data.add(s, visual_s, a, old_prob, r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis])

    @tf.function
    def _get_prob(self, s, visual_s, a):
        with tf.device(self.device):
            if self.action_type == 'continuous':
                mu, sigma = self.actor_net(s, visual_s)
                norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma + self.sigma_offset)
                prob = tf.reduce_mean(norm_dist.prob(a), axis=1, keepdims=True)
            else:
                action_probs = self.actor_net(s, visual_s)
                prob = tf.reduce_sum(tf.multiply(action_probs, a), axis=1, keepdims=True)
            return prob

    def no_op_store(self, s, visual_s, a, r, s_, visual_s_, done):
        old_prob = np.ones_like(r)
        assert isinstance(a, np.ndarray), "store_data need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store_data need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store_data need done type is np.ndarray"
        if self.policy_mode == 'OFF':
            self.data.add(s, visual_s, a, old_prob[:, np.newaxis], r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis])

    def learn(self, episode):
        s, visual_s, a, old_prob, r, s_, visual_s_, done = self.data.sample()
        if self.use_priority:
            self.IS_w = self.data.get_IS_w()
        actor_loss, critic_loss, entropy, td_error = self.train(s, visual_s, a, r, s_, visual_s_, done, old_prob)
        if self.use_priority:
            self.data.update(td_error, episode)
        tf.summary.experimental.set_step(self.global_step)
        if entropy is not None:
            tf.summary.scalar('LOSS/entropy', entropy)
        tf.summary.scalar('LOSS/actor_loss', actor_loss)
        tf.summary.scalar('LOSS/critic_loss', critic_loss)
        tf.summary.scalar('LEARNING_RATE/lr', self.lr)
        self.recorder.writer.flush()

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, r, s_, visual_s_, done, old_prob):
        done = tf.cast(done, tf.float64)
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                if self.action_type == 'continuous':
                    next_mu, _ = self.actor_net(s_, visual_s_)
                    max_q_next = tf.stop_gradient(self.critic_net(s_, visual_s_, next_mu))
                else:
                    _all_a = tf.expand_dims(tf.one_hot([i for i in range(self.a_counts)], self.a_counts), 1)
                    all_a = tf.reshape(tf.tile(_all_a, [1, tf.shape(a)[0], 1]), [-1, self.a_counts])
                    max_q_next = tf.stop_gradient(tf.reduce_max(
                        self.critic_net(tf.tile(s_, [self.a_counts, 1]), tf.tile(visual_s_, [self.a_counts, 1]), all_a),
                        axis=0, keepdims=True))
                q = self.critic_net(s, visual_s, a)
                td_error = q - (r + self.gamma * (1 - done) * max_q_next)
                critic_loss = tf.reduce_mean(tf.square(td_error) * self.IS_w)
            critic_grads = tape.gradient(critic_loss, self.critic_net.trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_net.trainable_variables)
            )
            with tf.GradientTape() as tape:
                if self.action_type == 'continuous':
                    mu, sigma = self.actor_net(s, visual_s)
                    norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma + self.sigma_offset)
                    prob = tf.reduce_mean(norm_dist.prob(a), axis=1, keepdims=True)
                    log_act_prob = norm_dist.log_prob(a)
                    entropy = tf.reduce_mean(norm_dist.entropy())
                else:
                    action_probs = self.actor_net(s, visual_s)
                    prob = tf.reduce_sum(tf.multiply(action_probs, a), axis=1, keepdims=True)
                    log_act_prob = tf.log(prob)
                q = self.critic_net(s, visual_s, a)
                ratio = tf.stop_gradient(prob / old_prob)
                q_value = tf.stop_gradient(q)
                actor_loss = -tf.reduce_mean(ratio * log_act_prob * q_value)
            actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return actor_loss, critic_loss, entropy if self.action_type == 'continuous' else None, td_error

    @tf.function(experimental_relax_shapes=True)
    def train_persistent(self, s, visual_s, a, r, s_, visual_s_, done, old_prob):
        done = tf.cast(done, tf.float64)
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                if self.action_type == 'continuous':
                    next_mu, _ = self.actor_net(s_, visual_s_)
                    max_q_next = tf.stop_gradient(self.critic_net(s_, visual_s_, next_mu))

                    mu, sigma = self.actor_net(s, visual_s)
                    norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma + self.sigma_offset)
                    prob = tf.reduce_mean(norm_dist.prob(a), axis=1, keepdims=True)
                    log_act_prob = norm_dist.log_prob(a)
                    entropy = tf.reduce_mean(norm_dist.entropy())
                else:
                    _all_a = tf.expand_dims(tf.one_hot([i for i in range(self.a_counts)], self.a_counts), 1)
                    all_a = tf.reshape(tf.tile(_all_a, [1, tf.shape(a)[0], 1]), [-1, self.a_counts])
                    max_q_next = tf.stop_gradient(tf.reduce_max(
                        self.critic_net(tf.tile(s_, [self.a_counts, 1]), tf.tile(visual_s_, [self.a_counts, 1]), all_a),
                        axis=0, keepdims=True))

                    action_probs = self.actor_net(s, visual_s)
                    prob = tf.reduce_sum(tf.multiply(action_probs, a), axis=1, keepdims=True)
                    log_act_prob = tf.log(prob)
                q = self.critic_net(s, visual_s, a)
                ratio = tf.stop_gradient(prob / old_prob)
                q_value = tf.stop_gradient(q)
                td_error = q - (r + self.gamma * (1 - done) * max_q_next)
                critic_loss = tf.reduce_mean(tf.square(td_error) * self.IS_w)
                actor_loss = -tf.reduce_mean(ratio * log_act_prob * q_value)
            critic_grads = tape.gradient(critic_loss, self.critic_net.trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_net.trainable_variables)
            )
            actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return actor_loss, critic_loss, entropy if self.action_type == 'continuous' else None, td_error
