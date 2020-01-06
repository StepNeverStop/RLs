import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import Nn
from utils.sth import sth
from Algorithms.tf2algos.base.off_policy import Off_Policy


class TD3(Off_Policy):
    '''
    Twin Delayed Deep Deterministic Policy Gradient, https://arxiv.org/abs/1802.09477
    '''
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,

                 ployak=0.995,
                 delay_num=2,
                 noise_type='gaussian',
                 share_visual_net=True,
                 gaussian_noise_sigma=0.2,
                 gaussian_noise_bound=0.2,
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 discrete_tau=1.0,
                 hidden_units={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'q': [32, 32]
                 },
                 **kwargs):
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim_or_list=a_dim_or_list,
            is_continuous=is_continuous,
            **kwargs)
        self.ployak = ployak
        self.delay_num = delay_num
        self.discrete_tau = discrete_tau
        self.gaussian_noise_sigma = gaussian_noise_sigma
        self.gaussian_noise_bound = gaussian_noise_bound

        self.share_visual_net = share_visual_net
        if self.share_visual_net:
            self.actor_visual_net = self.critic_visual_net = Nn.VisualNet('visual_net', self.visual_dim)
        else:
            self.actor_visual_net = Nn.VisualNet('actor_visual_net', self.visual_dim)
            self.critic_visual_net = Nn.VisualNet('critic_visual_net', self.visual_dim)

        if self.is_continuous:
            self.actor_net = Nn.actor_dpg(self.s_dim, self.a_counts, 'actor_net', hidden_units['actor_continuous'], visual_net=self.actor_visual_net)
            self.actor_target_net = Nn.actor_dpg(self.s_dim, self.a_counts, 'actor_target_net', hidden_units['actor_continuous'], visual_net=self.actor_visual_net)
            if noise_type == 'gaussian':
                self.action_noise = Nn.ClippedNormalActionNoise(mu=np.zeros(self.a_counts), sigma=self.gaussian_noise_sigma * np.ones(self.a_counts), bound=self.gaussian_noise_bound)
            elif noise_type == 'ou':
                self.action_noise = Nn.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_counts), sigma=0.2 * np.exp(-self.episode / 10) * np.ones(self.a_counts))
        else:
            self.actor_net = Nn.actor_discrete(self.s_dim, self.a_counts, 'actor_net', hidden_units['actor_discrete'], visual_net=self.actor_visual_net)
            self.actor_target_net = Nn.actor_discrete(self.s_dim, self.a_counts, 'actor_target_net', hidden_units['actor_discrete'], visual_net=self.actor_visual_net)
            self.gumbel_dist = tfp.distributions.Gumbel(0, 1)
        self.q1_net = Nn.critic_q_one(self.s_dim, self.a_counts, 'q1_net', hidden_units['q'], visual_net=self.critic_visual_net)
        self.q1_target_net = Nn.critic_q_one(self.s_dim, self.a_counts, 'q1_target_net', hidden_units['q'], visual_net=self.critic_visual_net)
        self.q2_net = Nn.critic_q_one(self.s_dim, self.a_counts, 'q2_net', hidden_units['q'], visual_net=self.critic_visual_net)
        self.q2_target_net = Nn.critic_q_one(self.s_dim, self.a_counts, 'q2_target_net', hidden_units['q'], visual_net=self.critic_visual_net)
        self.update_target_net_weights(
            self.actor_target_net.weights + self.q1_target_net.weights + self.q2_target_net.weights,
            self.actor_net.weights + self.q1_net.weights + self.q2_net.weights
        )
        self.actor_lr = tf.keras.optimizers.schedules.PolynomialDecay(actor_lr, self.max_episode, 1e-10, power=1.0)
        self.critic_lr = tf.keras.optimizers.schedules.PolynomialDecay(critic_lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.critic_lr(self.episode))
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.actor_lr(self.episode))

    def show_logo(self):
        self.recorder.logger.info('''
　　　ｘｘｘｘｘｘｘｘｘ　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　　　　　
　　　ｘｘ　　ｘ　　ｘｘ　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　ｘｘ　　　　　
　　　ｘｘ　　ｘ　　ｘｘ　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘｘ　ｘｘ　　　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　　　ｘｘｘ　　　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　　ｘｘｘｘ　　　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　　　　ｘｘｘ　　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　ｘｘｘ　　　　
　　　　　ｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　
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
                pi = tf.clip_by_value(mu + self.action_noise(), -1, 1)
            else:
                logits = self.actor_net(s, visual_s)
                mu = tf.argmax(logits, axis=1)
                cate_dist = tfp.distributions.Categorical(logits)
                pi = cate_dist.sample()
            if evaluation == True:
                return mu
            else:
                return pi

    def learn(self, **kwargs):
        self.episode = kwargs['episode']
        for i in range(kwargs['step']):
            if self.data.is_lg_batch_size:
                s, visual_s, a, r, s_, visual_s_, done = self.data.sample()
                if self.use_priority:
                    self.IS_w = self.data.get_IS_w()
                td_error, summaries = self.train(s, visual_s, a, r, s_, visual_s_, done)
                if self.use_priority:
                    td_error = np.squeeze(td_error.numpy())
                    self.data.update(td_error, self.episode)
                self.update_target_net_weights(
                    self.actor_target_net.weights + self.q1_target_net.weights + self.q2_target_net.weights,
                    self.actor_net.weights + self.q1_net.weights + self.q2_net.weights,
                    self.ployak)
                summaries.update(dict([
                    ['LEARNING_RATE/actor_lr', self.actor_lr(self.episode)],
                    ['LEARNING_RATE/critic_lr', self.critic_lr(self.episode)]
                ]))
                self.write_training_summaries(self.global_step, summaries)

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, r, s_, visual_s_, done):
        s, visual_s, a, r, s_, visual_s_, done = self.cast(s, visual_s, a, r, s_, visual_s_, done)
        with tf.device(self.device):
            for _ in range(self.delay_num):
                with tf.GradientTape() as tape:
                    if self.is_continuous:
                        target_mu = self.actor_target_net(s_, visual_s_)
                        action_target = tf.clip_by_value(target_mu + self.action_noise(), -1, 1)
                    else:
                        target_logits = self.actor_target_net(s_, visual_s_)
                        target_cate_dist = tfp.distributions.Categorical(target_logits)
                        target_pi = target_cate_dist.sample()
                        action_target = tf.one_hot(target_pi, self.a_counts, dtype=tf.float32)
                    q1 = self.q1_net(s, visual_s, a)
                    q1_target = self.q1_target_net(s_, visual_s_, action_target)
                    q2 = self.q2_net(s, visual_s, a)
                    q2_target = self.q2_target_net(s_, visual_s_, action_target)
                    q_target = tf.minimum(q1_target, q2_target)
                    dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                    td_error1 = q1 - dc_r
                    td_error2 = q2 - dc_r
                    q1_loss = tf.reduce_mean(tf.square(td_error1) * self.IS_w)
                    q2_loss = tf.reduce_mean(tf.square(td_error2) * self.IS_w)
                    critic_loss = 0.5 * (q1_loss + q2_loss)
                critic_grads = tape.gradient(critic_loss, self.q1_net.tv + self.q2_net.tv)
                self.optimizer_critic.apply_gradients(
                    zip(critic_grads, self.q1_net.tv + self.q2_net.tv)
                )
            with tf.GradientTape() as tape:
                if self.is_continuous:
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
                q1_actor = self.q1_net(s, visual_s, pi)
                actor_loss = -tf.reduce_mean(q1_actor)
            actor_grads = tape.gradient(actor_loss, self.actor_net.tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.tv)
            )
            self.global_step.assign_add(1)
            return td_error1 + td_error2 / 2, dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/q_min', tf.reduce_min(tf.minimum(q1, q2))],
                ['Statistics/q_mean', tf.reduce_mean(tf.minimum(q1, q2))],
                ['Statistics/q_max', tf.reduce_max(tf.maximum(q1, q2))],
            ])

    @tf.function(experimental_relax_shapes=True)
    def train_persistent(self, s, visual_s, a, r, s_, visual_s_, done):
        s, visual_s, a, r, s_, visual_s_, done = self.cast(s, visual_s, a, r, s_, visual_s_, done)
        with tf.device(self.device):
            for _ in range(2):
                with tf.GradientTape(persistent=True) as tape:
                    if self.is_continuous:
                        target_mu = self.actor_target_net(s_, visual_s_)
                        action_target = tf.clip_by_value(target_mu + self.action_noise(), -1, 1)
                        mu = self.actor_net(s, visual_s)
                        pi = tf.clip_by_value(mu + self.action_noise(), -1, 1)
                    else:
                        target_logits = self.actor_target_net(s_, visual_s_)
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
                    q1 = self.q1_net(s, visual_s, a)
                    q1_target = self.q1_target_net(s_, visual_s_, action_target)
                    q2 = self.q2_net(s, visual_s, a)
                    q2_target = self.q2_target_net(s_, visual_s_, action_target)
                    q1_actor = self.q1_net(s, visual_s, pi)
                    q_target = tf.minimum(q1_target, q2_target)
                    dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                    td_error1 = q1 - dc_r
                    td_error2 = q2 - dc_r
                    q1_loss = tf.reduce_mean(tf.square(td_error1) * self.IS_w)
                    q2_loss = tf.reduce_mean(tf.square(td_error2) * self.IS_w)
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
            return td_error1 + td_error2 / 2, dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/q_min', tf.reduce_min(tf.minimum(q1, q2))],
                ['Statistics/q_mean', tf.reduce_mean(tf.minimum(q1, q2))],
                ['Statistics/q_max', tf.reduce_max(tf.maximum(q1, q2))]
            ])
