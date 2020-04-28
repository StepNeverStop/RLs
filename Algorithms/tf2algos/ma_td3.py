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
                 n=1,
                 i=0,
                 hidden_units={
                     'actor': [32, 32],
                     'q': [32, 32]
                 },
                 **kwargs):
        assert is_continuous, 'matd3 only support continuous action space'
        raise Exception('MA系列存在问题，还未修复')
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

        # self.action_noise = Nn.NormalActionNoise(mu=np.zeros(self.a_counts), sigma=1 * np.ones(self.a_counts))
        self.action_noise = Nn.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_counts), sigma=0.2 * np.exp(-self.episode / 10) * np.ones(self.a_counts))

        _actor_net = lambda: Nn.actor_dpg(self.s_dim, 0, self.a_counts, hidden_units['actor'])
        self.actor_net = _actor_net()
        self.actor_target_net = _actor_net()
        _q_net = lambda: Nn.critic_q_one((self.s_dim) * self.n, 0, (self.a_counts) * self.n, hidden_units['q'])
        self.q1_net = _q_net()
        self.q1_target_net = _q_net()
        self.q2_net = _q_net()
        self.q2_target_net = _q_net()
        self.update_target_net_weights(
            self.actor_target_net.weights + self.q1_target_net.weights + self.q2_target_net.weights,
            self.actor_net.weights + self.q1_net.weights + self.q2_net.weights
        )
        self.actor_lr = self.init_lr(actor_lr)
        self.critic_lr = self.init_lr(critic_lr)
        self.optimizer_critic = self.init_optimizer(self.critic_lr)
        self.optimizer_actor = self.init_optimizer(self.actor_lr)

        self.model_recorder(dict(
            actor=self.actor_net,
            q1_net=self.q1_net,
            q2_net=self.q2_net,
            optimizer_critic=self.optimizer_critic,
            optimizer_actor=self.optimizer_actor
        ))

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
            mu = self.actor_net(vector_input)
            if evaluation == True:
                return mu
            else:
                return tf.clip_by_value(mu + self.action_noise(), -1, 1)

    @tf.function
    def _get_target_action(self, vector_input):
        vector_input = self.cast(vector_input)
        with tf.device(self.device):
            target_mu = self.actor_target_net(vector_input)
        return tf.clip_by_value(target_mu + self.action_noise(), -1, 1)

    def learn(self, episode, ap, al, ss, ss_, aa, aa_, s, r):
        self.episode = episode
        ap, al, ss, ss_, aa, aa_, s, r = map(self.data_convert, (ap, al, ss, ss_, aa, aa_, s, r))
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
        with tf.device(self.device):
            for _ in range(2):
                with tf.GradientTape() as tape:
                    q1 = self.q1_net(ss, aa)
                    q1_target = self.q1_target_net(ss_, aa_)
                    q2 = self.q2_net(ss, aa)
                    q2_target = self.q2_target_net(ss_, aa_)
                    q_target = tf.minimum(q1_target, q2_target)
                    dc_r = tf.stop_gradient(r + self.gamma * q_target)
                    td_error1 = q1 - dc_r
                    td_error2 = q2 - dc_r
                    q1_loss = tf.reduce_mean(tf.square(td_error1))
                    q2_loss = tf.reduce_mean(tf.square(td_error2))
                    critic_loss = 0.5 * (q1_loss + q2_loss)
                critic_grads = tape.gradient(critic_loss, self.q1_net.trainable_variables + self.q2_net.trainable_variables)
                self.optimizer_critic.apply_gradients(
                    zip(critic_grads, self.q1_net.trainable_variables + self.q2_net.trainable_variables)
                )
            with tf.GradientTape() as tape:
                mu = self.actor_net(s)
                mumu = tf.concat((q_actor_a_previous, mu, q_actor_a_later), axis=1)
                q1_actor = self.q1_net(ss, mumu)
                actor_loss = -tf.reduce_mean(q1_actor)
            actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.trainable_variables)
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
        with tf.device(self.device):
            for _ in range(2):
                with tf.GradientTape(persistent=True) as tape:
                    mu = self.actor_net(s)
                    mumu = tf.concat((q_actor_a_previous, mu, q_actor_a_later), axis=1)
                    q1 = self.q1_net(ss, aa)
                    q1_target = self.q1_target_net(ss_, aa_)
                    q2 = self.q2_net(ss, aa)
                    q2_target = self.q2_target_net(ss_, aa_)
                    q1_actor = self.q1_net(ss, mumu)
                    q_target = tf.minimum(q1_target, q2_target)
                    dc_r = tf.stop_gradient(r + self.gamma * q_target)
                    td_error1 = q1 - dc_r
                    td_error2 = q2 - dc_r
                    q1_loss = tf.reduce_mean(tf.square(td_error1))
                    q2_loss = tf.reduce_mean(tf.square(td_error2))
                    critic_loss = 0.5 * (q1_loss + q2_loss)
                    actor_loss = -tf.reduce_mean(q1_actor)
                critic_grads = tape.gradient(critic_loss, self.q1_net.trainable_variables + self.q2_net.trainable_variables)
                self.optimizer_critic.apply_gradients(
                    zip(critic_grads, self.q1_net.trainable_variables + self.q2_net.trainable_variables)
                )
            actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/q1_loss', q1_loss],
                ['LOSS/q2_loss', q2_loss],
                ['LOSS/critic_loss', critic_loss]
            ])
