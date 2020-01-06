import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import Nn
from utils.sth import sth
from utils.tf2_utils import get_TensorSpecs, gaussian_clip_rsample, gaussian_likelihood_sum, gaussian_entropy
from Algorithms.tf2algos.base.on_policy import On_Policy


class A2C(On_Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,

                 epoch=5,
                 beta=1.0e-3,
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
        self.beta = beta
        self.epoch = epoch
        self.share_visual_net = share_visual_net
        if self.share_visual_net:
            self.actor_visual_net = self.critic_visual_net = Nn.VisualNet('visual_net', self.visual_dim)
        else:
            self.actor_visual_net = Nn.VisualNet('actor_visual_net', self.visual_dim)
            self.critic_visual_net = Nn.VisualNet('critic_visual_net', self.visual_dim)

        self.TensorSpecs = get_TensorSpecs([self.s_dim], self.visual_dim, [self.a_counts], [1])
        if self.is_continuous:
            self.actor_net = Nn.actor_mu(self.s_dim, self.a_counts, 'actor_net', hidden_units['actor_continuous'], visual_net=self.actor_visual_net)
            self.log_std = tf.Variable(initial_value=-0.5 * np.ones(self.a_counts, dtype=np.float32), trainable=True)
            self.actor_net.tv += [self.log_std]
        else:
            self.actor_net = Nn.actor_discrete(self.s_dim, self.a_counts, 'actor_net', hidden_units['actor_discrete'], visual_net=self.actor_visual_net)
        self.critic_net = Nn.critic_v(self.s_dim, 'critic_net', hidden_units['critic'], visual_net=self.critic_visual_net)
        self.actor_lr = tf.keras.optimizers.schedules.PolynomialDecay(actor_lr, self.max_episode, 1e-10, power=1.0)
        self.critic_lr = tf.keras.optimizers.schedules.PolynomialDecay(critic_lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.critic_lr(self.episode))
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.actor_lr(self.episode))

    def show_logo(self):
        self.recorder.logger.info('''
　　　　　　　ｘｘ　　　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘ　　　　
　　　　　　ｘｘｘ　　　　　　　　　　　ｘｘ　ｘｘｘ　　　　　　　　ｘｘｘ　　ｘｘ　　　　
　　　　　　ｘｘｘ　　　　　　　　　　　ｘｘ　ｘｘｘ　　　　　　　　ｘｘ　　　　ｘｘ　　　
　　　　　　ｘ　ｘｘ　　　　　　　　　　　　　ｘｘ　　　　　　　　　ｘｘ　　　　　　　　　
　　　　　ｘｘ　ｘｘ　　　　　　　　　　　　ｘｘｘ　　　　　　　　ｘｘｘ　　　　　　　　　
　　　　　ｘｘｘｘｘｘ　　　　　　　　　　　ｘｘ　　　　　　　　　ｘｘｘ　　　　　　　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　　　ｘｘ　　　　　　　　　　　ｘｘ　　　　ｘｘ　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　　ｘｘ　　ｘ　　　　　　　　　ｘｘｘ　　ｘｘｘ　　　
　　　ｘｘｘ　　ｘｘｘｘｘ　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘ　　　　
        ''')

    def choose_action(self, s, visual_s, evaluation=False):
        a = self._get_action(s, visual_s, evaluation).numpy()
        return a if self.is_continuous else sth.int2action_index(a, self.a_dim_or_list)\

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

    def calculate_statistics(self):
        s, visual_s = self.data.s_.values[-1], self.data.visual_s_.values[-1]
        init_value = np.squeeze(self.critic_net(s, visual_s))
        self.data['discounted_reward'] = sth.discounted_sum(self.data.r.values, self.gamma, init_value, self.data.done.values)

    def get_sample_data(self, index):
        i_data = self.data.iloc[index:index + self.batch_size]
        s = np.vstack(i_data.s.values).astype(np.float32)
        visual_s = np.vstack(i_data.visual_s.values).astype(np.float32)
        a = np.vstack(i_data.a.values).astype(np.float32)
        dc_r = np.vstack(i_data.discounted_reward.values).reshape(-1, 1).astype(np.float32)
        return s, visual_s, a, dc_r

    def learn(self, **kwargs):
        assert self.batch_size <= self.data.shape[0], "batch_size must less than the length of an episode"
        self.episode = kwargs['episode']
        self.calculate_statistics()
        for _ in range(self.epoch):
            for index in range(0, self.data.shape[0], self.batch_size):
                s, visual_s, a, dc_r = map(tf.convert_to_tensor, self.get_sample_data(index))
                actor_loss, critic_loss, entropy = self.train.get_concrete_function(
                    *self.TensorSpecs)(s, visual_s, a, dc_r)
        self.global_step.assign_add(1)
        self.write_training_summaries(self.episode, dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/critic_loss', critic_loss],
            ['Statistics/entropy', entropy],
            ['LEARNING_RATE/actor_lr', self.actor_lr(self.episode)],
            ['LEARNING_RATE/critic_lr', self.critic_lr(self.episode)]
        ]))
        self.clear()

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, dc_r):
        s, visual_s, a, dc_r = self.cast(s, visual_s, a, dc_r)
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                v = self.critic_net(s, visual_s)
                td_error = dc_r - v
                critic_loss = tf.reduce_mean(tf.square(td_error))
            critic_grads = tape.gradient(critic_loss, self.critic_net.tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_net.tv)
            )
            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu = self.actor_net(s, visual_s)
                    log_act_prob = gaussian_likelihood_sum(mu, a, self.log_std)
                    entropy = gaussian_entropy(self.log_std)
                else:
                    logits = self.actor_net(s, visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    log_act_prob = tf.reduce_sum(a * logp_all, axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                v = self.critic_net(s, visual_s)
                advantage = tf.stop_gradient(dc_r - v)
                actor_loss = -(tf.reduce_mean(log_act_prob * advantage) + self.beta * entropy)
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
            return actor_loss, critic_loss, entropy

    @tf.function(experimental_relax_shapes=True)
    def train_persistent(self, s, visual_s, a, dc_r):
        s, visual_s, a, dc_r = self.cast(s, visual_s, a, dc_r)
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                if self.is_continuous:
                    mu = self.actor_net(s, visual_s)
                    log_act_prob = gaussian_likelihood_sum(mu, a, self.log_std)
                    entropy = gaussian_entropy(self.log_std)
                else:
                    logits = self.actor_net(s, visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    log_act_prob = tf.reduce_sum(a * logp_all, axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                v = self.critic_net(s, visual_s)
                advantage = tf.stop_gradient(dc_r - v)
                td_error = dc_r - v
                critic_loss = tf.reduce_mean(tf.square(td_error))
                actor_loss = -(tf.reduce_mean(log_act_prob * advantage) + self.beta * entropy)
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
            return actor_loss, critic_loss, entropy
