import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import Nn
from utils.sth import sth
from .policy import Policy


class DPG(Policy):
    # off-policy DPG
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
                 discrete_tau=1.0,
                 hidden_units={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'q': [32, 32]
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
        self.discrete_tau = discrete_tau
        if self.action_type == 'continuous':
            # self.action_noise = Nn.NormalActionNoise(mu=np.zeros(self.a_counts), sigma=1 * np.ones(self.a_counts))
            self.action_noise = Nn.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_counts), sigma=0.2 * np.exp(-self.episode / 10) * np.ones(self.a_counts))
            self.actor_net = Nn.actor_dpg(self.s_dim, self.visual_dim, self.a_counts, 'actor_net', hidden_units['actor_continuous'])
        else:
            self.actor_net = Nn.actor_discrete(self.s_dim, self.visual_dim, self.a_counts, 'actor_net', hidden_units['actor_discrete'])
            self.gumbel_dist = tfp.distributions.Gumbel(0, 1)
        self.q_net = Nn.critic_q_one(self.s_dim, self.visual_dim, self.a_counts, 'q_net', hidden_units['q'])
        self.actor_lr = tf.keras.optimizers.schedules.PolynomialDecay(actor_lr, self.max_episode, 1e-10, power=1.0)
        self.critic_lr = tf.keras.optimizers.schedules.PolynomialDecay(critic_lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.critic_lr(self.episode))
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.actor_lr(self.episode))
        self.generate_recorder(
            logger2file=logger2file,
            model=self
        )
        self.recorder.logger.info('''
　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　　　　　　　
　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘ　　　ｘｘｘｘｘ　　　
　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　ｘｘｘ　　　　
　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　
        ''')

    def choose_action(self, s, visual_s):
        a = self._get_action(s, visual_s)[-1].numpy()
        return a if self.action_type == 'continuous' else sth.int2action_index(a, self.a_dim_or_list)

    def choose_inference_action(self, s, visual_s):
        a = self._get_action(s, visual_s)[0].numpy()
        return a if self.action_type == 'continuous' else sth.int2action_index(a, self.a_dim_or_list)

    @tf.function
    def _get_action(self, vector_input, visual_input):
        with tf.device(self.device):
            if self.action_type == 'continuous':
                mu = self.actor_net(vector_input, visual_input)
                pi = tf.clip_by_value(mu + self.action_noise(), -1, 1)
            else:
                logits = self.actor_net(vector_input, visual_input)
                mu = tf.argmax(logits, axis=1)
                cate_dist = tfp.distributions.Categorical(logits)
                pi = cate_dist.sample()
        return mu, pi

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        self.off_store(s, visual_s, a, r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis])

    def learn(self, **kwargs):
        self.episode = kwargs['episode']
        self.recorder.writer.set_as_default()
        for i in range(kwargs['step']):
            if self.data.is_lg_batch_size:
                s, visual_s, a, r, s_, visual_s_, done = self.data.sample()
                if self.use_priority:
                    self.IS_w = self.data.get_IS_w()
                actor_loss, q_loss, td_error = self.train(s, visual_s, a, r, s_, visual_s_, done)
                if self.use_priority:
                    self.data.update(td_error, self.episode)
                tf.summary.experimental.set_step(self.global_step)
                tf.summary.scalar('LOSS/actor_loss', actor_loss)
                tf.summary.scalar('LOSS/critic_loss', q_loss)
                tf.summary.scalar('LEARNING_RATE/actor_lr', self.actor_lr(self.episode))
                tf.summary.scalar('LEARNING_RATE/critic_lr', self.critic_lr(self.episode))
                self.recorder.writer.flush()

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, r, s_, visual_s_, done):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                if self.action_type == 'continuous':
                    target_mu = self.actor_net(s_, visual_s_)
                    action_target = tf.clip_by_value(target_mu + self.action_noise(), -1, 1)
                else:
                    target_logits = self.actor_net(s_, visual_s_)
                    target_cate_dist = tfp.distributions.Categorical(target_logits)
                    target_pi = target_cate_dist.sample()
                    action_target = tf.one_hot(target_pi, self.a_counts, dtype=tf.float32)
                q_target = self.q_net(s_, visual_s_, action_target)
                dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                q = self.q_net(s, visual_s, a)
                td_error = q - dc_r
                q_loss = 0.5 * tf.reduce_mean(tf.square(td_error) * self.IS_w)
            q_grads = tape.gradient(q_loss, self.q_net.trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(q_grads, self.q_net.trainable_variables)
            )
            with tf.GradientTape() as tape:
                if self.action_type == 'continuous':
                    mu = self.actor_net(s, visual_s)
                    pi = tf.clip_by_value(mu + self.action_noise(), -1, 1)
                else:
                    logits = self.actor_net(s, visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    gumbel_noise = tf.cast(self.gumbel_dist.sample([a.shape[0], self.a_counts]), dtype=tf.float32)
                    _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_counts)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    pi = _pi_diff + _pi
                q_actor = self.q_net(s, visual_s, pi)
                actor_loss = -tf.reduce_mean(q_actor)
            actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return actor_loss, q_loss, td_error

    @tf.function(experimental_relax_shapes=True)
    def train_persistent(self, s, visual_s, a, r, s_, visual_s_, done):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                if self.action_type == 'continuous':
                    target_mu = self.actor_net(s_, visual_s_)
                    action_target = tf.clip_by_value(target_mu + self.action_noise(), -1, 1)
                    mu = self.actor_net(s, visual_s)
                    pi = tf.clip_by_value(mu + self.action_noise(), -1, 1)
                else:
                    target_logits = self.actor_net(s_, visual_s_)
                    target_cate_dist = tfp.distributions.Categorical(target_logits)
                    target_pi = target_cate_dist.sample()
                    action_target = tf.one_hot(target_pi, self.a_counts, dtype=tf.float32)
                    logits = self.actor_net(s, visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    gumbel_noise = tf.cast(self.gumbel_dist.sample([a.shape[0], self.a_counts]), dtype=tf.float32)
                    _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_counts)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    pi = _pi_diff + _pi
                q_target = self.q_net(s_, visual_s_, action_target)
                dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                q = self.q_net(s, visual_s, a)
                td_error = q - dc_r
                q_loss = 0.5 * tf.reduce_mean(tf.square(td_error) * self.IS_w)
                q_actor = self.q_net(s, visual_s, pi)
                actor_loss = -tf.reduce_mean(q_actor)
            q_grads = tape.gradient(q_loss, self.q_net.trainable_variables)
            self.optimizer_critic.apply_gradients(
                zip(q_grads, self.q_net.trainable_variables)
            )
            actor_grads = tape.gradient(actor_loss, self.actor_net.trainable_variables)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return actor_loss, q_loss, td_error
