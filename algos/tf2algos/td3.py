import rls
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from algos.tf2algos.base.off_policy import make_off_policy_class
from rls.modules import DoubleQ


class TD3(make_off_policy_class(mode='share')):
    '''
    Twin Delayed Deep Deterministic Policy Gradient, https://arxiv.org/abs/1802.09477
    '''

    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim,
                 is_continuous,

                 ployak=0.995,
                 delay_num=2,
                 noise_type='gaussian',
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
            a_dim=a_dim,
            is_continuous=is_continuous,
            **kwargs)
        self.ployak = ployak
        self.delay_num = delay_num
        self.discrete_tau = discrete_tau
        self.gaussian_noise_sigma = gaussian_noise_sigma
        self.gaussian_noise_bound = gaussian_noise_bound

        if self.is_continuous:
            def _actor_net(): return rls.actor_dpg(self.feat_dim, self.a_dim, hidden_units['actor_continuous'])
            if noise_type == 'gaussian':
                self.action_noise = rls.ClippedNormalActionNoise(mu=np.zeros(self.a_dim), sigma=self.gaussian_noise_sigma * np.ones(self.a_dim), bound=self.gaussian_noise_bound)
            elif noise_type == 'ou':
                self.action_noise = rls.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_dim), sigma=0.2 * np.ones(self.a_dim))
        else:
            def _actor_net(): return rls.actor_discrete(self.feat_dim, self.a_dim, hidden_units['actor_discrete'])
            self.gumbel_dist = tfp.distributions.Gumbel(0, 1)

        self.actor_net = _actor_net()
        self.actor_target_net = _actor_net()
        self.actor_tv = self.actor_net.trainable_variables

        def _q_net(): return rls.critic_q_one(self.feat_dim, self.a_dim, hidden_units['q'])
        self.critic_net = DoubleQ(_q_net)
        self.critic_target_net = DoubleQ(_q_net)
        self.critic_tv = self.critic_net.trainable_variables + self.other_tv

        self.update_target_net_weights(
            self.actor_target_net.weights + self.critic_target_net.weights,
            self.actor_net.weights + self.critic_net.weights
        )
        self.actor_lr, self.critic_lr = map(self.init_lr, [actor_lr, critic_lr])
        self.optimizer_actor, self.optimizer_critic = map(self.init_optimizer, [self.actor_lr, self.critic_lr])

        self.model_recorder(dict(
            actor=self.actor_net,
            critic_net=self.critic_net,
            optimizer_actor=self.optimizer_actor,
            optimizer_critic=self.optimizer_critic
        ))

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
        mu, pi, self.cell_state = self._get_action(s, visual_s, self.cell_state)
        a = mu.numpy() if evaluation else pi.numpy()
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state):
        with tf.device(self.device):
            feat, cell_state = self.get_feature(s, visual_s, cell_state=cell_state, record_cs=True)
            if self.is_continuous:
                mu = self.actor_net(feat)
                pi = tf.clip_by_value(mu + self.action_noise(), -1, 1)
            else:
                logits = self.actor_net(feat)
                mu = tf.argmax(logits, axis=1)
                cate_dist = tfp.distributions.Categorical(logits)
                pi = cate_dist.sample()
            return mu, pi, cell_state

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'train_function': self.train,
                'update_function': lambda: self.update_target_net_weights(
                    self.actor_target_net.weights + self.critic_target_net.weights,
                    self.actor_net.weights + self.critic_net.weights,
                    self.ployak),
                'summary_dict': dict([
                    ['LEARNING_RATE/actor_lr', self.actor_lr(self.train_step)],
                    ['LEARNING_RATE/critic_lr', self.critic_lr(self.train_step)]
                ])
            })

    @tf.function(experimental_relax_shapes=True)
    def train(self, memories, isw, crsty_loss, cell_state):
        ss, vvss, a, r, done = memories
        batch_size = tf.shape(a)[0]
        with tf.device(self.device):
            for _ in range(self.delay_num):
                with tf.GradientTape() as tape:
                    feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                    if self.is_continuous:
                        target_mu = self.actor_target_net(feat_)
                        action_target = tf.clip_by_value(target_mu + self.action_noise(), -1, 1)
                    else:
                        target_logits = self.actor_target_net(feat_)
                        logp_all = tf.nn.log_softmax(target_logits)
                        gumbel_noise = tf.cast(self.gumbel_dist.sample([batch_size, self.a_dim]), dtype=tf.float32)
                        _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                        _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_dim)
                        _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                        action_target = _pi_diff + _pi
                    q1, q2 = self.critic_net(feat, a)
                    q_target = self.critic_target_net.get_min(feat_, action_target)
                    dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                    td_error1 = q1 - dc_r
                    td_error2 = q2 - dc_r
                    q1_loss = tf.reduce_mean(tf.square(td_error1) * isw)
                    q2_loss = tf.reduce_mean(tf.square(td_error2) * isw)
                    critic_loss = 0.5 * (q1_loss + q2_loss) + crsty_loss
                critic_grads = tape.gradient(critic_loss, self.critic_tv)
                self.optimizer_critic.apply_gradients(
                    zip(critic_grads, self.critic_tv)
                )
            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu = self.actor_net(feat)
                else:
                    logits = self.actor_net(feat)
                    _pi = tf.nn.softmax(logits)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(logits, axis=-1), self.a_dim, dtype=tf.float32)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    mu = _pi_diff + _pi
                q1_actor = self.critic_net.Q1(feat, mu)
                actor_loss = -tf.reduce_mean(q1_actor)
            actor_grads = tape.gradient(actor_loss, self.actor_tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_tv)
            )
            self.global_step.assign_add(1)
            return (td_error1 + td_error2) / 2, dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/q_min', tf.reduce_min(tf.minimum(q1, q2))],
                ['Statistics/q_mean', tf.reduce_mean(tf.minimum(q1, q2))],
                ['Statistics/q_max', tf.reduce_max(tf.maximum(q1, q2))],
            ])

    @tf.function(experimental_relax_shapes=True)
    def train_persistent(self, memories, isw, crsty_loss, cell_state):
        ss, vvss, a, r, done = memories
        batch_size = tf.shape(a)[0]
        with tf.device(self.device):
            for _ in range(2):
                with tf.GradientTape(persistent=True) as tape:
                    feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                    if self.is_continuous:
                        target_mu = self.actor_target_net(feat_)
                        action_target = tf.clip_by_value(target_mu + self.action_noise(), -1, 1)
                        mu = self.actor_net(feat)
                    else:
                        target_logits = self.actor_target_net(feat_)
                        logp_all = tf.nn.log_softmax(target_logits)
                        gumbel_noise = tf.cast(self.gumbel_dist.sample([batch_size, self.a_dim]), dtype=tf.float32)
                        _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                        _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_dim)
                        _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                        action_target = _pi_diff + _pi
                        logits = self.actor_net(feat)
                        _pi = tf.nn.softmax(logits)
                        _pi_true_one_hot = tf.one_hot(tf.argmax(logits, axis=-1), self.a_dim, dtype=tf.float32)
                        _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                        mu = _pi_diff + _pi
                    q1, q2 = self.critic_net(feat, a)
                    q_target = self.critic_target_net.get_min(feat_, action_target)
                    q1_actor = self.critic_net.Q1(feat, mu)
                    dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                    td_error1 = q1 - dc_r
                    td_error2 = q2 - dc_r
                    q1_loss = tf.reduce_mean(tf.square(td_error1) * isw)
                    q2_loss = tf.reduce_mean(tf.square(td_error2) * isw)
                    critic_loss = 0.5 * (q1_loss + q2_loss) + crsty_loss
                    actor_loss = -tf.reduce_mean(q1_actor)
                critic_grads = tape.gradient(critic_loss, self.critic_tv)
                self.optimizer_critic.apply_gradients(
                    zip(critic_grads, self.critic_tv)
                )
            actor_grads = tape.gradient(actor_loss, self.actor_tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_tv)
            )
            self.global_step.assign_add(1)
            return (td_error1 + td_error2) / 2, dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/q_min', tf.reduce_min(tf.minimum(q1, q2))],
                ['Statistics/q_mean', tf.reduce_mean(tf.minimum(q1, q2))],
                ['Statistics/q_max', tf.reduce_max(tf.maximum(q1, q2))]
            ])
