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

        if self.is_continuous:
            self.actor_net = Nn.actor_mu(self.rnn_net.hdim, self.a_counts, hidden_units['actor_continuous'])
            self.log_std = tf.Variable(initial_value=-0.5 * np.ones(self.a_counts, dtype=np.float32), trainable=True)
            self.actor_tv = self.actor_net.trainable_variables + [self.log_std]
        else:
            self.actor_net = Nn.actor_discrete(self.rnn_net.hdim, self.a_counts, hidden_units['actor_discrete'])
            self.actor_tv = self.actor_net.trainable_variables
        self.critic_net = Nn.critic_q_one(self.rnn_net.hdim, self.a_counts, hidden_units['critic'])
        self.critic_tv = self.critic_net.trainable_variables + self.other_tv
        self.actor_lr = tf.keras.optimizers.schedules.PolynomialDecay(actor_lr, self.max_episode, 1e-10, power=1.0)
        self.critic_lr = tf.keras.optimizers.schedules.PolynomialDecay(critic_lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.critic_lr(self.episode))
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.actor_lr(self.episode))

        self.model_recorder(dict(
            actor=self.actor_net,
            critic=self.critic_net,
            optimizer_actor=self.optimizer_actor,
            optimizer_critic=self.optimizer_critic
        ))
    
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
        feat, self.cell_state = self.get_feature(s, visual_s, self.cell_state, record_cs=True, train=False)
        a, _lp = self._get_action(feat)
        a = a.numpy()
        self._log_prob = _lp.numpy()
        return a if self.is_continuous else sth.int2action_index(a, self.a_dim_or_list)

    @tf.function
    def _get_action(self, feat):
        with tf.device(self.device):
            if self.is_continuous:
                mu = self.actor_net(feat)
                sample_op, _ = gaussian_clip_rsample(mu, self.log_std)
                log_prob = gaussian_likelihood_sum(mu, sample_op, self.log_std)
            else:
                logits = self.actor_net(feat)
                norm_dist = tfp.distributions.Categorical(logits)
                sample_op = norm_dist.sample()
                log_prob = norm_dist.log_prob(sample_op)
        return sample_op, log_prob

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "store_data need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store_data need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store_data need done type is np.ndarray"
        if not self.is_continuous:
            a = sth.action_index2one_hot(a, self.a_dim_or_list)
        old_log_prob = self._log_prob
        self.data.add(s, visual_s, a, old_log_prob,
                      r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis])

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
            self._learn(function_dict={
                'train_function': self.train,
                'update_function': lambda : None,
                'summary_dict': dict([
                                ['LEARNING_RATE/actor_lr', self.actor_lr(self.episode)],
                                ['LEARNING_RATE/critic_lr', self.critic_lr(self.episode)]
                            ]),
                'data_list': ['s', 'visual_s', 'a', 'old_log_prob', 'r', 's_', 'visual_s_', 'done']
            })

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, old_log_prob, r, s_, visual_s_, done):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat = self.get_feature(s, visual_s)
                feat_ = self.get_feature(s_, visual_s_)
                if self.is_continuous:
                    next_mu = self.actor_net(feat_)
                    max_q_next = tf.stop_gradient(self.critic_net(feat_, next_mu))
                else:
                    logits = self.actor_net(feat_)
                    max_a = tf.argmax(logits, axis=1)
                    max_a_one_hot = tf.one_hot(max_a, self.a_counts, dtype=tf.float32)
                    max_q_next = tf.stop_gradient(self.critic_net(feat_, max_a_one_hot))
                q = self.critic_net(feat, a)
                td_error = q - (r + self.gamma * (1 - done) * max_q_next)
                critic_loss = tf.reduce_mean(tf.square(td_error) * self.IS_w)
            critic_grads = tape.gradient(critic_loss, self.critic_tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_tv)
            )
            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu = self.actor_net(feat)
                    log_prob = gaussian_likelihood_sum(mu, a, self.log_std)
                    entropy = gaussian_entropy(self.log_std)
                else:
                    logits = self.actor_net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    log_prob = tf.reduce_sum(tf.multiply(logp_all, a), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                q = self.critic_net(feat, a)
                ratio = tf.stop_gradient(tf.exp(log_prob - old_log_prob))
                q_value = tf.stop_gradient(q)
                actor_loss = -tf.reduce_mean(ratio * log_prob * q_value)
            actor_grads = tape.gradient(actor_loss, self.actor_tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_tv)
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
    def train_persistent(self, s, visual_s, a, old_log_prob, r, s_, visual_s_, done):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                feat = self.get_feature(s, visual_s)
                feat_ = self.get_feature(s_, visual_s_)
                if self.is_continuous:
                    next_mu = self.actor_net(feat_)
                    max_q_next = tf.stop_gradient(self.critic_net(feat_, next_mu))
                    mu, sigma = self.actor_net(feat)
                    log_prob = gaussian_likelihood_sum(mu, a, self.log_std)
                    entropy = gaussian_entropy(self.log_std)
                else:
                    logits = self.actor_net(feat_)
                    max_a = tf.argmax(logits, axis=1)
                    max_a_one_hot = tf.one_hot(max_a, self.a_counts)
                    max_q_next = tf.stop_gradient(self.critic_net(feat_, max_a_one_hot))
                    logits = self.actor_net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    log_prob = tf.reduce_sum(tf.multiply(logp_all, a), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                q = self.critic_net(feat, a)
                ratio = tf.stop_gradient(tf.exp(log_prob - old_log_prob))
                q_value = tf.stop_gradient(q)
                td_error = q - (r + self.gamma * (1 - done) * max_q_next)
                critic_loss = tf.reduce_mean(tf.square(td_error) * self.IS_w)
                actor_loss = -tf.reduce_mean(ratio * log_prob * q_value)
            critic_grads = tape.gradient(critic_loss, self.critic_tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_tv)
            )
            actor_grads = tape.gradient(actor_loss, self.actor_tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_tv)
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
