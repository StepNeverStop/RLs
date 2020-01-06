import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import Nn
from utils.sth import sth
from utils.tf2_utils import gaussian_clip_rsample, gaussian_likelihood_sum, gaussian_entropy
from Algorithms.tf2algos.base.off_policy import Off_Policy


class AC(Off_Policy):
    # off-policy actor-critic
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,

                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 share_visual_net=True,
                 hidden_units={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'critic': [32, 32]
                 },
                 **kwargs):
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim_or_list=a_dim_or_list,
            is_continuous=is_continuous,
            **kwargs)
        self.share_visual_net = share_visual_net
        if self.share_visual_net:
            self.actor_visual_net = self.critic_visual_net = Nn.VisualNet('visual_net', self.visual_dim)
        else:
            self.actor_visual_net = Nn.VisualNet('actor_visual_net', self.visual_dim)
            self.critic_visual_net = Nn.VisualNet('critic_visual_net', self.visual_dim)
        if self.is_continuous:
            self.actor_net = Nn.actor_mu(self.s_dim, self.a_counts, 'actor_net', hidden_units['actor_continuous'], visual_net=self.actor_visual_net)
            self.log_std = tf.Variable(initial_value=-0.5 * np.ones(self.a_counts, dtype=np.float32), trainable=True)
            self.actor_net.tv += [self.log_std]
        else:
            self.actor_net = Nn.actor_discrete(self.s_dim, self.a_counts, 'actor_net', hidden_units['actor_discrete'], visual_net=self.actor_visual_net)
        self.critic_net = Nn.critic_q_one(self.s_dim, self.a_counts, 'critic_net', hidden_units['critic'], visual_net=self.critic_visual_net)
        self.actor_lr = tf.keras.optimizers.schedules.PolynomialDecay(actor_lr, self.max_episode, 1e-10, power=1.0)
        self.critic_lr = tf.keras.optimizers.schedules.PolynomialDecay(critic_lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.critic_lr(self.episode))
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.actor_lr(self.episode))
    
    def show_logo(self):
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

    def choose_action(self, s, visual_s, evaluation=False):
        a = self._get_action(s, visual_s, evaluation).numpy()
        return a if self.is_continuous else sth.int2action_index(a, self.a_dim_or_list)

    @tf.function
    def _get_action(self, s, visual_s, evaluation):
        s, visual_s = self.cast(s, visual_s)
        with tf.device(self.device):
            if self.is_continuous:
                mu = self.actor_net(s, visual_s)
                sample_op, _ = gaussian_clip_rsample(mu, self.log_std)
            else:
                logits = self.actor_net(s, visual_s)
                norm_dist = tfp.distributions.Categorical(logits)
                sample_op = norm_dist.sample()
        return sample_op

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "store_data need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store_data need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store_data need done type is np.ndarray"
        if not self.is_continuous:
            a = sth.action_index2one_hot(a, self.a_dim_or_list)
        old_log_prob = self._get_log_prob(s, visual_s, a).numpy()
        self.data.add(s, visual_s, a, old_log_prob,
                      r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis])

    @tf.function
    def _get_log_prob(self, s, visual_s, a):
        s, visual_s, a = self.cast(s, visual_s, a)
        with tf.device(self.device):
            if self.is_continuous:
                mu = self.actor_net(s, visual_s)
                log_prob = gaussian_likelihood_sum(mu, a, self.log_std)
            else:
                logits = self.actor_net(s, visual_s)
                logp_all = tf.nn.log_softmax(logits)
                log_prob = tf.reduce_sum(tf.multiply(logp_all, a), axis=1, keepdims=True)
            return log_prob

    def no_op_store(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "store_data need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store_data need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store_data need done type is np.ndarray"
        old_log_prob = np.ones_like(r)
        if not self.is_continuous:
            a = sth.action_index2one_hot(a, self.a_dim_or_list)
        self.data.add(s, visual_s, a, old_log_prob[:, np.newaxis], r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis])

    def learn(self, **kwargs):
        self.episode = kwargs['episode']
        for i in range(kwargs['step']):
            s, visual_s, a, old_log_prob, r, s_, visual_s_, done = self.data.sample()
            if self.use_priority:
                self.IS_w = self.data.get_IS_w()
            td_error, summaries = self.train(s, visual_s, a, r, s_, visual_s_, done, old_log_prob)
            if self.use_priority:
                td_error = np.squeeze(td_error.numpy())
                self.data.update(td_error, self.episode)
            summaries.update(dict([
                ['LEARNING_RATE/actor_lr', self.actor_lr(self.episode)],
                ['LEARNING_RATE/critic_lr', self.critic_lr(self.episode)]
            ]))
            self.write_training_summaries(self.global_step, summaries)

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, r, s_, visual_s_, done, old_log_prob):
        s, visual_s, a, r, s_, visual_s_, done, old_log_prob = self.cast(s, visual_s, a, r, s_, visual_s_, done, old_log_prob)
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                if self.is_continuous:
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
            critic_grads = tape.gradient(critic_loss, self.critic_net.tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_net.tv)
            )
            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu = self.actor_net(s, visual_s)
                    log_prob = gaussian_likelihood_sum(mu, a, self.log_std)
                    entropy = gaussian_entropy(self.log_std)
                else:
                    logits = self.actor_net(s, visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    log_prob = tf.reduce_sum(tf.multiply(logp_all, a), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                q = self.critic_net(s, visual_s, a)
                ratio = tf.stop_gradient(tf.exp(log_prob - old_log_prob))
                q_value = tf.stop_gradient(q)
                actor_loss = -tf.reduce_mean(ratio * log_prob * q_value)
            if self.is_continuous:
                actor_grads = tape.gradient(actor_loss, self.actor_net.tv)
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_net.tv)
                )
            else:
                actor_grads = tape.gradient(actor_loss, self.actor_net.tv)
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_net.tv)
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
        s, visual_s, a, r, s_, visual_s_, done, old_log_prob = self.cast(s, visual_s, a, r, s_, visual_s_, done, old_log_prob)
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                if self.is_continuous:
                    next_mu = self.actor_net(s_, visual_s_)
                    max_q_next = tf.stop_gradient(self.critic_net(s_, visual_s_, next_mu))
                    mu, sigma = self.actor_net(s, visual_s)
                    log_prob = gaussian_likelihood_sum(mu, a, self.log_std)
                    entropy = gaussian_entropy(self.log_std)
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
            critic_grads = tape.gradient(critic_loss, self.critic_net.tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_net.tv)
            )
            if self.is_continuous:
                actor_grads = tape.gradient(actor_loss, self.actor_net.tv)
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_net.tv)
                )
            else:
                actor_grads = tape.gradient(actor_loss, self.actor_net.tv)
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_net.tv)
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
