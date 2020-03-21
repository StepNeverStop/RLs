import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import Nn
from utils.sth import sth
from utils.tf2_utils import clip_nn_log_std, squash_rsample, gaussian_entropy
from Algorithms.tf2algos.base.off_policy import Off_Policy
from utils.sundry_utils import LinearAnnealing


class SAC_V(Off_Policy):
    """
        Soft Actor Critic with Value neural network. https://arxiv.org/abs/1812.05905
        Soft Actor-Critic for Discrete Action Settings. https://arxiv.org/abs/1910.07207
    """

    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,

                 alpha=0.2,
                 annealing=True,
                 last_alpha=0.01,
                 ployak=0.995,
                 use_gumbel=True,
                 discrete_tau=1.0,
                 log_std_bound=[-20, 2],
                 share_visual_net=True,
                 hidden_units={
                     'actor_continuous': {
                         'share': [128, 128],
                         'mu': [64],
                         'log_std': [64]
                     },
                     'actor_discrete': [64, 32],
                     'q': [128, 128],
                     'v': [128, 128]
                 },
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 alpha_lr=5.0e-4,
                 auto_adaption=True,
                 **kwargs):
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim_or_list=a_dim_or_list,
            is_continuous=is_continuous,
            **kwargs)
        self.ployak = ployak
        self.use_gumbel = use_gumbel
        self.discrete_tau = discrete_tau
        self.log_std_min, self.log_std_max = log_std_bound[:]
        self.auto_adaption = auto_adaption
        self.annealing = annealing

        if self.auto_adaption:
            self.log_alpha = tf.Variable(initial_value=0.0, name='log_alpha', dtype=tf.float32, trainable=True)
        else:
            self.log_alpha = tf.Variable(initial_value=tf.math.log(alpha), name='log_alpha', dtype=tf.float32, trainable=False)
            if self.annealing:
                self.alpha_annealing = LinearAnnealing(alpha, last_alpha, 1e6)

        self.share_visual_net = share_visual_net
        if self.share_visual_net:
            self.actor_visual_net = self.q_visual_net = self.v_visual_net = self._visual_net()
        else:
            self.actor_visual_net = self._visual_net()
            self.q_visual_net = self._visual_net()
            self.v_visual_net = self._visual_net()

        rnn_net = self._rnn_net(self.actor_visual_net.hdim)

        if self.is_continuous:
            actor_net = Nn.actor_continuous(rnn_net.hdim, self.a_counts, hidden_units['actor_continuous'])
        else:
            actor_net = Nn.actor_discrete(rnn_net.hdim, self.a_counts, hidden_units['actor_discrete'])
            if self.use_gumbel:
                self.gumbel_dist = tfp.distributions.Gumbel(0, 1)
        
        if self.is_continuous or self.use_gumbel:
            critic_net = Nn.critic_q_one
        else:
            critic_net = Nn.critic_q_all

        self.actor_net = Nn.VisualObsRNN(
            net=actor_net,
            visual_net=self.actor_visual_net,
            rnn_net=rnn_net,
            rnn_net_grad=False
        )
        self.q1_net = Nn.VisualObsRNN(
            net=critic_net(rnn_net.hdim, self.a_counts, hidden_units['q']),
            visual_net=self.q_visual_net,
            rnn_net=rnn_net
        )
        self.q2_net = Nn.VisualObsRNN(
            net=critic_net(rnn_net.hdim, self.a_counts, hidden_units['q']),
            visual_net=self.q_visual_net,
            rnn_net=rnn_net
        )
        self.v_net = Nn.VisualObsRNN(
            net=critic_net(rnn_net.hdim, hidden_units['v']),
            visual_net=self.v_visual_net,
            rnn_net=rnn_net
        )
        self.v_target_net = Nn.VisualObsRNN(
            net=critic_net(rnn_net.hdim, hidden_units['v']),
            visual_net=self.v_visual_net,
            rnn_net=rnn_net
        )

        self.update_target_net_weights(self.v_target_net.uv, self.v_net.uv)
        self.actor_lr = tf.keras.optimizers.schedules.PolynomialDecay(actor_lr, self.max_episode, 1e-10, power=1.0)
        self.critic_lr = tf.keras.optimizers.schedules.PolynomialDecay(critic_lr, self.max_episode, 1e-10, power=1.0)
        self.alpha_lr = tf.keras.optimizers.schedules.PolynomialDecay(alpha_lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.critic_lr(self.episode))
        self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.actor_lr(self.episode))
        self.optimizer_alpha = tf.keras.optimizers.Adam(learning_rate=self.alpha_lr(self.episode))

        self.model_recorder(dict(
            actor=self.actor_net,
            q1_net=self.q1_net,
            q2_net=self.q2_net,
            v_net=self.v_net,
            optimizer_actor=self.optimizer_actor,
            optimizer_critic=self.optimizer_critic,
            optimizer_alpha=self.optimizer_alpha,
            ))

    def show_logo(self):
        self.recorder.logger.info('''
　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　　ｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘ　　ｘｘｘ　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　　　ｘｘｘ　　　　　　　　　　ｘｘｘ　　ｘｘ　　　　　　　　　　　　　　　　　　　　　　ｘｘｘ　　　　ｘ　　　　
　　　　ｘｘ　　　　ｘ　　　　　　　　　　ｘｘｘ　　　　　　　　　　ｘｘ　　　　ｘｘ　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　ｘｘ　　　　
　　　　ｘｘｘｘ　　　　　　　　　　　　　ｘ　ｘｘ　　　　　　　　　ｘｘ　　　　　　　　　　　ｘｘ　　ｘｘ　　ｘｘ　　　　　　　ｘｘｘ　　ｘｘ　　　　
　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘｘ　ｘｘ　　　　　　　　ｘｘｘ　　　　　　　　　　　ｘｘ　　ｘｘ　　ｘｘ　　　　　　　　ｘｘ　ｘｘ　　　　　
　　　　　　　　ｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　　　　　　　　　　ｘｘ　　ｘｘ　　ｘｘ　　　　　　　　ｘｘｘｘｘ　　　　　
　　　　ｘ　　　　ｘｘ　　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘ　　　　ｘｘ　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘ　　　　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘｘ　　ｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘ　　　　　　
　　　　ｘｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　ｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　ｘ　　　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　
        ''')

    def choose_action(self, s, visual_s, evaluation=False):
        a = self._get_action(s, visual_s, evaluation).numpy()
        return a if self.is_continuous else sth.int2action_index(a, self.a_dim_or_list)

    @tf.function
    def _get_action(self, s, visual_s, evaluation):
        s, visual_s = self.cast(s, visual_s)
        with tf.device(self.device):
            if self.is_continuous:
                mu, log_std = self.actor_net.choose(s, visual_s)
                log_std = clip_nn_log_std(log_std, self.log_std_min, self.log_std_max)
                pi, _ = squash_rsample(mu, log_std)
                mu = tf.tanh(mu)    # squash mu
            else:
                logits = self.actor_net.choose(s, visual_s)
                mu = tf.argmax(logits, axis=1)
                cate_dist = tfp.distributions.Categorical(logits)
                pi = cate_dist.sample()
            if evaluation == True:
                return mu
            else:
                return pi

    def learn(self, **kwargs):
        self.episode = kwargs['episode']
        def _train(s, visual_s, a, r, s_, visual_s_, done):
            if self.is_continuous or self.use_gumbel:
                td_error, summaries = self.train(s, visual_s, a, r, s_, visual_s_, done)
            else:
                td_error, summaries = self.train_discrete(s, visual_s, a, r, s_, visual_s_, done)
            if self.annealing and not self.auto_adaption:
                self.log_alpha.assign(tf.math.log(tf.cast(self.alpha_annealing(self.global_step.numpy()), tf.float32)))
            return td_error, summaries

        for i in range(kwargs['step']):
            self._learn(function_dict={
                'train_function': _train,
                'update_function': lambda : self.update_target_net_weights(self.v_target_net.uv, self.v_net.uv, self.ployak),
                'summary_dict': dict([
                                    ['LEARNING_RATE/actor_lr', self.actor_lr(self.episode)],
                                    ['LEARNING_RATE/critic_lr', self.critic_lr(self.episode)],
                                    ['LEARNING_RATE/alpha_lr', self.alpha_lr(self.episode)]
                                ])
            })

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, r, s_, visual_s_, done):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu, log_std = self.actor_net(s, visual_s)
                    log_std = clip_nn_log_std(log_std, self.log_std_min, self.log_std_max)
                    pi, log_pi = squash_rsample(mu, log_std)
                    entropy = gaussian_entropy(log_std)
                else:
                    logits = self.actor_net(s, visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    gumbel_noise = tf.cast(self.gumbel_dist.sample([a.shape[0], self.a_counts]), dtype=tf.float32)
                    _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_counts)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    pi = _pi_diff + _pi
                    log_pi = tf.reduce_sum(tf.multiply(logp_all, pi), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                q1_pi = self.q1_net(s, visual_s, pi)
                actor_loss = -tf.reduce_mean(q1_pi - tf.exp(self.log_alpha) * log_pi)
            actor_grads = tape.gradient(actor_loss, self.actor_net.tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.tv)
            )
            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu, log_std = self.actor_net(s, visual_s)
                    log_std = clip_nn_log_std(log_std, self.log_std_min, self.log_std_max)
                    pi, log_pi = squash_rsample(mu, log_std)
                else:
                    logits = self.actor_net(s, visual_s)
                    cate_dist = tfp.distributions.Categorical(logits)
                    pi = cate_dist.sample()
                    log_pi = cate_dist.log_prob(pi)
                    pi = tf.one_hot(pi, self.a_counts, dtype=tf.float32)
                q1 = self.q1_net(s, visual_s, a)
                q2 = self.q2_net(s, visual_s, a)
                v = self.v_net(s, visual_s)
                q1_pi = self.q1_net(s, visual_s, pi)
                q2_pi = self.q2_net(s, visual_s, pi)
                v_target = self.v_target_net(s_, visual_s_)
                dc_r = tf.stop_gradient(r + self.gamma * v_target * (1 - done))
                v_from_q_stop = tf.stop_gradient(tf.minimum(q1_pi, q2_pi) - tf.exp(self.log_alpha) * log_pi)
                td_v = v - v_from_q_stop
                td_error1 = q1 - dc_r
                td_error2 = q2 - dc_r
                q1_loss = tf.reduce_mean(tf.square(td_error1) * self.IS_w)
                q2_loss = tf.reduce_mean(tf.square(td_error2) * self.IS_w)
                v_loss_stop = tf.reduce_mean(tf.square(td_v) * self.IS_w)
                critic_loss = 0.5 * q1_loss + 0.5 * q2_loss + 0.5 * v_loss_stop
            critic_grads = tape.gradient(critic_loss, self.q1_net.tv + self.q2_net.tv + self.v_net.tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.q1_net.tv + self.q2_net.tv + self.v_net.tv)
            )
            if self.auto_adaption:
                with tf.GradientTape() as tape:
                    if self.is_continuous:
                        mu, log_std = self.actor_net(s, visual_s)
                        log_std = clip_nn_log_std(log_std, self.log_std_min, self.log_std_max)
                        # pi, log_pi = squash_rsample(mu, log_std)
                        norm_dist = tfp.distributions.Normal(loc=mu, scale=tf.exp(log_std))
                        log_pi = tf.reduce_sum(norm_dist.log_prob(norm_dist.sample()),axis=-1)
                    else:
                        logits = self.actor_net(s, visual_s)
                        cate_dist = tfp.distributions.Categorical(logits)
                        log_pi = cate_dist.log_prob(cate_dist.sample())
                    alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(log_pi - self.a_counts))
                alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
                self.optimizer_alpha.apply_gradients(
                    zip(alpha_grads, [self.log_alpha])
                )
            self.global_step.assign_add(1)
            summaries = dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/q1_loss', q1_loss],
                ['LOSS/q2_loss', q2_loss],
                ['LOSS/v_loss', v_loss_stop],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/log_alpha', self.log_alpha],
                ['Statistics/alpha', tf.exp(self.log_alpha)],
                ['Statistics/entropy', entropy],
                ['Statistics/q_min', tf.reduce_min(tf.minimum(q1, q2))],
                ['Statistics/q_mean', tf.reduce_mean(tf.minimum(q1, q2))],
                ['Statistics/q_max', tf.reduce_max(tf.maximum(q1, q2))],
                ['Statistics/v_mean', tf.reduce_mean(v)]
            ])
            if self.auto_adaption:
                summaries.update({
                    'LOSS/alpha_loss': alpha_loss
                })
            return td_error1 + td_error2 / 2, summaries

    @tf.function(experimental_relax_shapes=True)
    def train_persistent(self, s, visual_s, a, r, s_, visual_s_, done):
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                if self.is_continuous:
                    mu, log_std = self.actor_net(s, visual_s)
                    log_std = clip_nn_log_std(log_std, self.log_std_min, self.log_std_max)
                    pi, log_pi = squash_rsample(mu, log_std)
                    entropy = gaussian_entropy(log_std)
                else:
                    logits = self.actor_net(s, visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    gumbel_noise = tf.cast(self.gumbel_dist.sample([a.shape[0], self.a_counts]), dtype=tf.float32)
                    _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_counts)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    pi = _pi_diff + _pi
                    log_pi = tf.reduce_sum(tf.multiply(logp_all, pi), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                q1 = self.q1_net(s, visual_s, a)
                q2 = self.q2_net(s, visual_s, a)
                v = self.v_net(s, visual_s)
                q1_pi = self.q1_net(s, visual_s, pi)
                q2_pi = self.q2_net(s, visual_s, pi)
                v_target = self.v_target_net(s_, visual_s_)
                dc_r = tf.stop_gradient(r + self.gamma * v_target * (1 - done))
                v_from_q_stop = tf.stop_gradient(tf.minimum(q1_pi, q2_pi) - tf.exp(self.log_alpha) * log_pi)
                td_v = v - v_from_q_stop
                td_error1 = q1 - dc_r
                td_error2 = q2 - dc_r
                q1_loss = tf.reduce_mean(tf.square(td_error1) * self.IS_w)
                q2_loss = tf.reduce_mean(tf.square(td_error2) * self.IS_w)
                v_loss_stop = tf.reduce_mean(tf.square(td_v) * self.IS_w)
                critic_loss = 0.5 * q1_loss + 0.5 * q2_loss + 0.5 * v_loss_stop
                actor_loss = -tf.reduce_mean(q1_pi - tf.exp(self.log_alpha) * log_pi)
                if self.auto_adaption:
                    alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(log_pi - self.a_counts))
            actor_grads = tape.gradient(actor_loss, self.actor_net.tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.tv)
            )
            critic_grads = tape.gradient(critic_loss, self.q1_net.tv + self.q2_net.tv + self.v_net.tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.q1_net.tv + self.q2_net.tv + self.v_net.tv)
            )
            if self.auto_adaption:
                alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
                self.optimizer_alpha.apply_gradients(
                    zip(alpha_grads, [self.log_alpha])
                )
            self.global_step.assign_add(1)
            summaries = dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/q1_loss', q1_loss],
                ['LOSS/q2_loss', q2_loss],
                ['LOSS/v_loss', v_loss_stop],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/log_alpha', self.log_alpha],
                ['Statistics/alpha', tf.exp(self.log_alpha)],
                ['Statistics/entropy', entropy],
                ['Statistics/q_min', tf.reduce_min(tf.minimum(q1, q2))],
                ['Statistics/q_mean', tf.reduce_mean(tf.minimum(q1, q2))],
                ['Statistics/q_max', tf.reduce_max(tf.maximum(q1, q2))],
                ['Statistics/v_mean', tf.reduce_mean(v)]
            ])
            if self.auto_adaption:
                summaries.update({
                    'LOSS/alpha_loss': alpha_loss
                })
            return td_error1 + td_error2 / 2, summaries
    
    @tf.function(experimental_relax_shapes=True)
    def train_discrete(self, s, visual_s, a, r, s_, visual_s_, done):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                logits = self.actor_net(s, visual_s)
                logp_all = tf.nn.log_softmax(logits)
                entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                q1_all = self.q1_net(s, visual_s)   # [B, A]
                q2_all = self.q2_net(s, visual_s)   # [B, A]
                q_all = tf.minimum(q1_all, q2_all)  # [B, A]
                actor_loss = -tf.reduce_mean(
                    tf.reduce_sum((q_all - tf.exp(self.log_alpha) * logp_all) * tf.exp(logp_all)) # [B, A] => [B,]
                )
            actor_grads = tape.gradient(actor_loss, self.actor_net.tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net.tv)
            )
            with tf.GradientTape() as tape:
                q1_all = self.q1_net(s, visual_s)   # [B, A]
                q2_all = self.q2_net(s, visual_s)   # [B, A]
                q_function = lambda x: tf.reduce_sum(x*a, axis=-1, keepdims=True)   #[B, 1]
                q1 = q_function(q1_all)
                q2 = q_function(q2_all)
                logits = self.actor_net(s, visual_s)   #[B, A]
                logp_all = tf.nn.log_softmax(logits)     #[B, A]
                v = self.v_net(s, visual_s) # [B, 1]
                v_target = self.v_target_net(s_, visual_s_) # [B, 1]
                dc_r = tf.stop_gradient(r + self.gamma * v_target * (1 - done))
                td_v = v - tf.stop_gradient(tf.minimum(
                    tf.reduce_sum(tf.exp(logp_all)*q1_all,axis=-1), 
                    tf.reduce_sum(tf.exp(logp_all)*q2_all,axis=-1)
                    ))
                td_error1 = q1 - dc_r
                td_error2 = q2 - dc_r
                q1_loss = tf.reduce_mean(tf.square(td_error1) * self.IS_w)
                q2_loss = tf.reduce_mean(tf.square(td_error2) * self.IS_w)
                v_loss_stop = tf.reduce_mean(tf.square(td_v) * self.IS_w)
                critic_loss = 0.5 * q1_loss + 0.5 * q2_loss + 0.5 * v_loss_stop
            critic_grads = tape.gradient(critic_loss, self.q1_net.tv + self.q2_net.tv + self.v_net.tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.q1_net.tv + self.q2_net.tv + self.v_net.tv)
            )
            if self.auto_adaption:
                with tf.GradientTape() as tape:
                    logits = self.actor_net(s, visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    corr = tf.stop_gradient(tf.reduce_sum((logp_all - self.a_counts) * tf.exp(logp_all), axis=-1))    #[B, A] => [B,]
                    alpha_loss = -tf.reduce_mean(self.log_alpha * corr)
                alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
                self.optimizer_alpha.apply_gradients(
                    zip(alpha_grads, [self.log_alpha])
                )
            self.global_step.assign_add(1)
            summaries = dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/q1_loss', q1_loss],
                ['LOSS/q2_loss', q2_loss],
                ['LOSS/v_loss', v_loss_stop],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/log_alpha', self.log_alpha],
                ['Statistics/alpha', tf.exp(self.log_alpha)],
                ['Statistics/entropy', entropy],
                ['Statistics/v_mean', tf.reduce_mean(v)]
            ])
            if self.auto_adaption:
                summaries.update({
                    'LOSS/alpha_loss': alpha_loss
                })
            return td_error1 + td_error2 / 2, summaries
