import numpy as np
import tensorflow as tf
import Nn
from Algorithms.tf2algos.base.policy import Policy


class MATD3(Policy):
    def __init__(self,
                 s_dim,
                 a_dim_or_list,
                 is_continuous,

                 ployak=0.995,
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 share_visual_net=True,
                 n=1,
                 i=0,
                 hidden_units={
                     'actor': [32, 32],
                     'q': [32, 32]
                 },
                 **kwargs):
        assert is_continuous, 'matd3 only support continuous action space'
        super().__init__(
            s_dim=s_dim,
            visual_sources=0,
            visual_resolution=0,
            a_dim_or_list=a_dim_or_list,
            is_continuous=is_continuous,
            **kwargs)
        self.n = n
        self.i = i
        self.ployak = ployak

        self.share_visual_net = share_visual_net
        if self.share_visual_net:
            self.actor_visual_net = self.critic_visual_net = Nn.VisualNet('visual_net', self.visual_dim)
        else:
            self.actor_visual_net = Nn.VisualNet('actor_visual_net', self.visual_dim)
            self.critic_visual_net = Nn.VisualNet('critic_visual_net', self.visual_dim)

        # self.action_noise = Nn.NormalActionNoise(mu=np.zeros(self.a_counts), sigma=1 * np.ones(self.a_counts))
        self.action_noise = Nn.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_counts), sigma=0.2 * np.exp(-self.episode / 10) * np.ones(self.a_counts))
        self.actor_net = Nn.actor_dpg(self.s_dim, 0, self.a_counts, 'actor_net', hidden_units['actor'], visual_net=self.actor_visual_net)
        self.actor_target_net = Nn.actor_dpg(self.s_dim, 0, self.a_counts, 'actor_target_net', hidden_units['actor'], visual_net=self.actor_visual_net)
        self.q1_net = Nn.critic_q_one((self.s_dim) * self.n, 0, (self.a_counts) * self.n, 'q1_net', hidden_units['q'], visual_net=self.critic_visual_net)
        self.q1_target_net = Nn.critic_q_one((self.s_dim) * self.n, 0, (self.a_counts) * self.n, 'q1_target_net', hidden_units['q'], visual_net=self.critic_visual_net)
        self.q2_net = Nn.critic_q_one((self.s_dim) * self.n, 0, (self.a_counts) * self.n, 'q2_net', hidden_units['q'], visual_net=self.critic_visual_net)
        self.q2_target_net = Nn.critic_q_one((self.s_dim) * self.n, 0, (self.a_counts) * self.n, 'q2_target_net', hidden_units['q'], visual_net=self.critic_visual_net)
        self.update_target_net_weights(
            self.actor_target_net.weights + self.q1_target_net.weights + self.q2_target_net.weights,
            self.actor_net.weights + self.q1_net.weights + self.q2_net.weights
        )
        self.actor_lr = tf.keras.optimizers.schedules.PolynomialDecay(actor_lr, self.max_episode, 1e-10, power=1.0)
        self.critic_lr = tf.keras.optimizers.schedules.PolynomialDecay(critic_lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.critic_lr(self.episode))
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.actor_lr(self.episode))
        self.recorder.logger.info(self.action_noise)

    def show_logo(self):
        self.recorder.logger.info('''
　　ｘｘｘｘ　　　　ｘｘｘ　　　　　　　　　ｘｘ　　　　　　　　　ｘｘｘｘｘｘｘｘｘ　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　　　　　
　　　ｘｘｘ　　　　ｘｘ　　　　　　　　　ｘｘｘ　　　　　　　　　ｘｘ　　ｘ　　ｘｘ　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　ｘｘ　　　　　
　　　　ｘｘｘ　　ｘｘｘ　　　　　　　　　ｘｘｘ　　　　　　　　　ｘｘ　　ｘ　　ｘｘ　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘｘ　ｘｘ　　　　　
　　　　ｘｘｘ　　ｘｘｘ　　　　　　　　　ｘ　ｘｘ　　　　　　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　　　ｘｘｘ　　　　　
　　　　ｘｘｘｘ　ｘ　ｘ　　　　　　　　ｘｘ　ｘｘ　　　　　　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　　ｘｘｘｘ　　　　　
　　　　ｘ　ｘｘｘｘ　ｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　　　　ｘｘｘ　　　　
　　　　ｘ　ｘｘｘ　　ｘ　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　
　　　　ｘ　　ｘｘ　　ｘ　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　　　　ｘ　　　　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　ｘｘｘ　　　　
　　ｘｘｘｘ　ｘｘｘｘｘｘ　　　　　ｘｘｘ　　ｘｘｘｘｘ　　　　　　　ｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　
        ''')

    def choose_action(self, s, evaluation=False):
        return self._get_action(s, evaluation).numpy()

    def get_target_action(self, s):
        return self._get_target_action(s).numpy()

    @tf.function
    def _get_action(self, vector_input, evaluation):
        vector_input = self.cast(vector_input)
        with tf.device(self.device):
            mu = self.actor_net(vector_input, None)
            if evaluation == True:
                return mu
            else:
                return tf.clip_by_value(mu + self.action_noise(), -1, 1)

    @tf.function
    def _get_target_action(self, vector_input):
        vector_input = self.cast(vector_input)
        with tf.device(self.device):
            target_mu = self.actor_target_net(vector_input, None)
        return tf.clip_by_value(target_mu + self.action_noise(), -1, 1)

    def learn(self, episode, ap, al, ss, ss_, aa, aa_, s, r):
        self.episode = episode
        summaries = self.train(ap, al, ss, ss_, aa, aa_, s, r)
        self.update_target_net_weights(
            self.actor_target_net.weights + self.q1_target_net.weights + self.q2_target_net.weights,
            self.actor_net.weights + self.q1_net.weights + self.q2_net.weights,
            self.ployak)
        summaries.update(dict([
            ['LEARNING_RATE/actor_lr', self.actor_lr(self.episode)],
            ['LEARNING_RATE/critic_lr', self.critic_lr(self.episode)]
        ]))
        self.write_training_summaries(self.global_step, summaries)

    def get_max_episode(self):
        """
        get the max episode of this training model.
        """
        return self.max_episode

    @tf.function(experimental_relax_shapes=True)
    def train(self, q_actor_a_previous, q_actor_a_later, ss, ss_, aa, aa_, s, r):
        q_actor_a_previous, q_actor_a_later, ss, ss_, aa, aa_, s, r = self.cast(q_actor_a_previous, q_actor_a_later, ss, ss_, aa, aa_, s, r)
        with tf.device(self.device):
            for _ in range(2):
                with tf.GradientTape() as tape:
                    q1 = self.q1_net(ss, None, aa)
                    q1_target = self.q1_target_net(ss_, None, aa_)
                    q2 = self.q2_net(ss, None, aa)
                    q2_target = self.q2_target_net(ss_, None, aa_)
                    q_target = tf.minimum(q1_target, q2_target)
                    dc_r = tf.stop_gradient(r + self.gamma * q_target)
                    td_error1 = q1 - dc_r
                    td_error2 = q2 - dc_r
                    q1_loss = tf.reduce_mean(tf.square(td_error1))
                    q2_loss = tf.reduce_mean(tf.square(td_error2))
                    critic_loss = 0.5 * (q1_loss + q2_loss)
                critic_grads = tape.gradient(critic_loss, self.q1_net.tv + self.q2_net.tv)
                self.optimizer_critic.apply_gradients(
                    zip(critic_grads, self.q1_net.tv + self.q2_net.tv)
                )
            with tf.GradientTape() as tape:
                mu = self.actor_net(s, None)
                mumu = tf.concat((q_actor_a_previous, mu, q_actor_a_later), axis=1)
                q1_actor = self.q1_net(ss, None, mumu)
                actor_loss = -tf.reduce_mean(q1_actor)
            actor_grads = tape.gradient(actor_loss, self.actor_net.tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.tv)
            )
            self.global_step.assign_add(1)
            return dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/q1_loss', q1_loss],
                ['LOSS/q2_loss', q2_loss],
                ['LOSS/critic_loss', critic_loss]
            ])

    @tf.function(experimental_relax_shapes=True)
    def train_persistent(self, q_actor_a_previous, q_actor_a_later, ss, ss_, aa, aa_, s, r):
        q_actor_a_previous, q_actor_a_later, ss, ss_, aa, aa_, s, r = self.cast(q_actor_a_previous, q_actor_a_later, ss, ss_, aa, aa_, s, r)
        with tf.device(self.device):
            for _ in range(2):
                with tf.GradientTape(persistent=True) as tape:
                    mu = self.actor_net(s, None)
                    mumu = tf.concat((q_actor_a_previous, mu, q_actor_a_later), axis=1)
                    q1 = self.q1_net(ss, None, aa)
                    q1_target = self.q1_target_net(ss_, None, aa_)
                    q2 = self.q2_net(ss, None, aa)
                    q2_target = self.q2_target_net(ss_, None, aa_)
                    q1_actor = self.q1_net(ss, None, mumu)
                    q_target = tf.minimum(q1_target, q2_target)
                    dc_r = tf.stop_gradient(r + self.gamma * q_target)
                    td_error1 = q1 - dc_r
                    td_error2 = q2 - dc_r
                    q1_loss = tf.reduce_mean(tf.square(td_error1))
                    q2_loss = tf.reduce_mean(tf.square(td_error2))
                    critic_loss = 0.5 * (q1_loss + q2_loss)
                    actor_loss = -tf.reduce_mean(q1_actor)
                critic_grads = tape.gradient(critic_loss, self.q1_net.tv + self.q2_net.tv)
                self.optimizer_critic.apply_gradients(
                    zip(critic_grads, self.q1_net.tv + self.q2_net.tv)
                )
            actor_grads = tape.gradient(actor_loss, self.actor_net.tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.tv)
            )
            self.global_step.assign_add(1)
            return dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/q1_loss', q1_loss],
                ['LOSS/q2_loss', q2_loss],
                ['LOSS/critic_loss', critic_loss]
            ])
