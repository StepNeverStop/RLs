import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import Nn
from utils.sth import sth
from utils.tf2_utils import gaussian_clip_reparam_sample, gaussian_likelihood, gaussian_entropy
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

                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 hidden_units={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'critic': [32, 32]
                 },
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
        if self.action_type == 'continuous':
            self.actor_net = Nn.actor_mu(self.s_dim, self.visual_dim, self.a_counts, 'actor_net', hidden_units['actor_continuous'])
            self.log_std = tf.Variable(initial_value=-0.5 * np.ones(self.a_counts, dtype=np.float32), trainable=True)
        else:
            self.actor_net = Nn.actor_discrete(self.s_dim, self.visual_dim, self.a_counts, 'actor_net', hidden_units['actor_discrete'])
        self.critic_net = Nn.critic_q_one(self.s_dim, self.visual_dim, self.a_counts, 'critic_net', hidden_units['critic'])
        self.actor_lr = tf.keras.optimizers.schedules.PolynomialDecay(actor_lr, self.max_episode, 1e-10, power=1.0)
        self.critic_lr = tf.keras.optimizers.schedules.PolynomialDecay(critic_lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.critic_lr(self.episode))
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.actor_lr(self.episode))
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
        a = self._get_action(s, visual_s).numpy()
        return a if self.action_type == 'continuous' else sth.int2action_index(a, self.a_dim_or_list)

    def choose_inference_action(self, s, visual_s):
        a = self._get_action(s, visual_s).numpy()
        return a if self.action_type == 'continuous' else sth.int2action_index(a, self.a_dim_or_list)

    @tf.function
    def _get_action(self, vector_input, visual_input):
        with tf.device(self.device):
            if self.action_type == 'continuous':
                mu = self.actor_net(vector_input, visual_input)
                sample_op, _ = gaussian_clip_reparam_sample(mu, self.log_std)
            else:
                logits = self.actor_net(vector_input, visual_input)
                norm_dist = tfp.distributions.Categorical(logits)
                sample_op = norm_dist.sample()
        return sample_op

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "store_data need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store_data need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store_data need done type is np.ndarray"
        if not self.action_type == 'continuous':
            a = sth.action_index2one_hot(a, self.a_dim_or_list)
        old_log_prob = self._get_log_prob(s, visual_s, a).numpy()
        self.data.add(s.astype(np.float32), visual_s.astype(np.float32), a.astype(np.float32), old_log_prob.astype(np.float32),
                      r[:, np.newaxis].astype(np.float32), s_.astype(np.float32), visual_s_.astype(np.float32), done[:, np.newaxis].astype(np.float32))

    @tf.function
    def _get_log_prob(self, s, visual_s, a):
        a = tf.cast(a, tf.float32)
        with tf.device(self.device):
            if self.action_type == 'continuous':
                mu = self.actor_net(s, visual_s)
                log_prob = gaussian_likelihood(mu, a, self.log_std)
            else:
                logits = self.actor_net(s, visual_s)
                logp_all = tf.nn.log_softmax(logits)
                log_prob = tf.reduce_sum(tf.multiply(logp_all, a), axis=1, keepdims=True)
            return log_prob

    def no_op_store(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "store_data need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store_data need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store_data need done type is np.ndarray"
        if self.policy_mode == 'OFF':
            old_log_prob = np.ones_like(r)
            if not self.action_type == 'continuous':
                a = sth.action_index2one_hot(a, self.a_dim_or_list)
            self.data.add(s.astype(np.float32), visual_s.astype(np.float32), a.astype(np.float32), old_log_prob[:, np.newaxis].astype(
                np.float32), r[:, np.newaxis].astype(np.float32), s_.astype(np.float32), visual_s_.astype(np.float32), done[:, np.newaxis].astype(np.float32))

    def learn(self, **kwargs):
        self.episode = kwargs['episode']
        self.recorder.writer.set_as_default()
        for i in range(kwargs['step']):
            s, visual_s, a, old_log_prob, r, s_, visual_s_, done = self.data.sample()
            if self.use_priority:
                self.IS_w = self.data.get_IS_w()
            td_error, summaries = self.train(s, visual_s, a, r, s_, visual_s_, done, old_log_prob)
            if self.use_priority:
                self.data.update(td_error, self.episode)
            tf.summary.experimental.set_step(self.global_step)
            self.write_training_summaries(summaries)
            tf.summary.scalar('LEARNING_RATE/actor_lr', self.actor_lr(self.episode))
            tf.summary.scalar('LEARNING_RATE/critic_lr', self.critic_lr(self.episode))
            self.recorder.writer.flush()

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, r, s_, visual_s_, done, old_log_prob):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                if self.action_type == 'continuous':
                    next_mu = self.actor_net(s_, visual_s_)
                    max_q_next = tf.stop_gradient(self.critic_net(s_, visual_s_, next_mu))
                else:
                    logits = self.actor_net(s_, visual_s_)
                    max_a = tf.argmax(logits, axis=1)
                    max_a_one_hot = tf.one_hot(max_a, self.a_counts, dtype=tf.float32)
                    max_q_next = tf.stop_gradient(self.critic_net(s_, visual_s_, max_a_one_hot))
                q = self.critic_net(s, visual_s, a)
                td_error = q - (r + self.gamma * (1 - done) * max_q_next)
                critic_loss = tf.reduce_mean(tf.square(td_error) * self.IS_w)
            critic_grads = tape.gradient(critic_loss, self.critic_net.trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_net.trainable_variables)
            )
            with tf.GradientTape() as tape:
                if self.action_type == 'continuous':
                    mu = self.actor_net(s, visual_s)
                    log_prob = gaussian_likelihood(mu, a, self.log_std)
                    entropy =  gaussian_entropy(self.log_std)
                else:
                    logits = self.actor_net(s, visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    log_prob = tf.reduce_sum(tf.multiply(logp_all, a), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                q = self.critic_net(s, visual_s, a)
                ratio = tf.stop_gradient(tf.exp(log_prob - old_log_prob))
                q_value = tf.stop_gradient(q)
                actor_loss = -tf.reduce_mean(ratio * log_prob * q_value)
            if self.action_type == 'continuous':
                actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables + [self.log_std])
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_net.trainable_variables + [self.log_std])
                )
            else:
                actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_net.trainable_variables)
                )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/q_max', tf.reduce_max(q)],
                ['Statistics/q_min', tf.reduce_min(q)],
                ['Statistics/q_mean', tf.reduce_mean(q)],
                ['Statistics/ratio', tf.reduce_mean(ratio)],
                ['Statistics/entropy', entropy]
            ])

    @tf.function(experimental_relax_shapes=True)
    def train_persistent(self, s, visual_s, a, r, s_, visual_s_, done, old_log_prob):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                if self.action_type == 'continuous':
                    next_mu = self.actor_net(s_, visual_s_)
                    max_q_next = tf.stop_gradient(self.critic_net(s_, visual_s_, next_mu))
                    mu, sigma = self.actor_net(s, visual_s)
                    log_prob = gaussian_likelihood(mu, a, self.log_std)
                    entropy =  gaussian_entropy(self.log_std)
                else:
                    logits = self.actor_net(s_, visual_s_)
                    max_a = tf.argmax(logits, axis=1)
                    max_a_one_hot = tf.one_hot(max_a, self.a_counts)
                    max_q_next = tf.stop_gradient(self.critic_net(s_, visual_s_, max_a_one_hot))
                    logits = self.actor_net(s, visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    log_prob = tf.reduce_sum(tf.multiply(logp_all, a), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                q = self.critic_net(s, visual_s, a)
                ratio = tf.stop_gradient(tf.exp(log_prob - old_log_prob))
                q_value = tf.stop_gradient(q)
                td_error = q - (r + self.gamma * (1 - done) * max_q_next)
                critic_loss = tf.reduce_mean(tf.square(td_error) * self.IS_w)
                actor_loss = -tf.reduce_mean(ratio * log_prob * q_value)
            critic_grads = tape.gradient(critic_loss, self.critic_net.trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_net.trainable_variables)
            )
            if self.action_type == 'continuous':
                actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables + [self.log_std])
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_net.trainable_variables + [self.log_std])
                )
            else:
                actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_net.trainable_variables)
                )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/q_max', tf.reduce_max(q)],
                ['Statistics/q_min', tf.reduce_min(q)],
                ['Statistics/q_mean', tf.reduce_mean(q)],
                ['Statistics/ratio', tf.reduce_mean(ratio)],
                ['Statistics/entropy', entropy]
            ])
