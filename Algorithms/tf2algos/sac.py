import Nn
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from utils.tf2_utils import clip_nn_log_std, squash_rsample, gaussian_entropy
from Algorithms.tf2algos.base.off_policy import Off_Policy
from utils.sundry_utils import LinearAnnealing


class SAC(Off_Policy):
    """
        Soft Actor Critic without Value neural network. https://arxiv.org/abs/1812.05905
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
                 hidden_units={
                     'actor_continuous': {
                         'share': [128, 128],
                         'mu': [64],
                         'log_std': [64]
                     },
                     'actor_discrete': [64, 32],
                     'q': [128, 128]
                 },
                 auto_adaption=True,
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 alpha_lr=5.0e-4,
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

        if self.is_continuous:
            self.actor_net = Nn.actor_continuous(self.rnn_net.hdim, self.a_counts, hidden_units['actor_continuous'])
        else:
            self.actor_net = Nn.actor_discrete(self.rnn_net.hdim, self.a_counts, hidden_units['actor_discrete'])
            if self.use_gumbel:
                self.gumbel_dist = tfp.distributions.Gumbel(0, 1)
        self.actor_tv = self.actor_net.trainable_variables
        
        if self.is_continuous or self.use_gumbel:
            critic_net = Nn.critic_q_one
        else:
            critic_net = Nn.critic_q_all
        
        _q_net = lambda : critic_net(self.rnn_net.hdim, self.a_counts, hidden_units['q'])
        self.q1_net = _q_net()
        self.q2_net = _q_net()
        self.q1_target_net = _q_net()
        self.q2_target_net = _q_net()
        self.critic_tv = self.q1_net.trainable_variables + self.q2_net.trainable_variables + self.other_tv
        
        self.update_target_net_weights(
            self.q1_target_net.weights + self.q2_target_net.weights,
            self.q1_net.weights + self.q2_net.weights
        )
        self.actor_lr, self.critic_lr, self.alpha_lr = map(self.init_lr, [actor_lr, critic_lr, alpha_lr])
        self.optimizer_actor, self.optimizer_critic, self.optimizer_alpha = map(self.init_optimizer, [self.actor_lr, self.critic_lr, self.alpha_lr])
        
        self.model_recorder(dict(
            actor=self.actor_net,
            q1_net=self.q1_net,
            q2_net=self.q2_net,
            optimizer_actor=self.optimizer_actor,
            optimizer_critic=self.optimizer_critic,
            optimizer_alpha=self.optimizer_alpha,
            ))
    
    def show_logo(self):
        self.recorder.logger.info('''
　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　　ｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　　　ｘｘｘ　　　　　　　　　　ｘｘｘ　　ｘｘ　　　　
　　　　ｘｘ　　　　ｘ　　　　　　　　　　ｘｘｘ　　　　　　　　　　ｘｘ　　　　ｘｘ　　　
　　　　ｘｘｘｘ　　　　　　　　　　　　　ｘ　ｘｘ　　　　　　　　　ｘｘ　　　　　　　　　
　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘｘ　ｘｘ　　　　　　　　ｘｘｘ　　　　　　　　　
　　　　　　　　ｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　　　　　　　　
　　　　ｘ　　　　ｘｘ　　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘ　　　　ｘｘ　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘｘ　　ｘｘｘ　　　
　　　　ｘｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　ｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘ　　
        ''')

    def choose_action(self, s, visual_s, evaluation=False):
        mu, pi, self.cell_state = self._get_action(s, visual_s, self.cell_state)
        a = mu.numpy() if evaluation else pi.numpy()
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state):
        with tf.device(self.device):
            feat, cell_state = self.get_feature(s, visual_s, cell_state=cell_state, record_cs=True, train=False)
            if self.is_continuous:
                mu, log_std = self.actor_net(feat)
                log_std = clip_nn_log_std(log_std, self.log_std_min, self.log_std_max)
                pi, _ = squash_rsample(mu, log_std)
                mu = tf.tanh(mu)  # squash mu
            else:
                logits = self.actor_net(feat)
                mu = tf.argmax(logits, axis=1)
                cate_dist = tfp.distributions.Categorical(logits)
                pi = cate_dist.sample()
            return mu, pi, cell_state

    def learn(self, **kwargs):
        self.episode = kwargs['episode']
        def _train(memories, isw, crsty_loss, cell_state):
            if self.is_continuous or self.use_gumbel:
                td_error, summaries = self.train_persistent(memories, isw, crsty_loss, cell_state)
            else:
                td_error, summaries = self.train_discrete(memories, isw, crsty_loss, cell_state)
            if self.annealing and not self.auto_adaption:
                self.log_alpha.assign(tf.math.log(tf.cast(self.alpha_annealing(self.global_step.numpy()), tf.float32)))
            return td_error, summaries

        for i in range(kwargs['step']):
            self._learn(function_dict={
                'train_function': _train,
                'update_function': lambda : self.update_target_net_weights(
                                            self.q1_target_net.weights + self.q2_target_net.weights,
                                            self.q1_net.weights + self.q2_net.weights,
                                            self.ployak),
                'summary_dict': dict([
                                ['LEARNING_RATE/actor_lr', self.actor_lr(self.episode)],
                                ['LEARNING_RATE/critic_lr', self.critic_lr(self.episode)],
                                ['LEARNING_RATE/alpha_lr', self.alpha_lr(self.episode)]
                                ])
            })

    @tf.function(experimental_relax_shapes=True)
    def train(self, memories, isw, crsty_loss, cell_state):
        ss, vvss, a, r, done = memories
        batch_size = tf.shape(a)[0]
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                if self.is_continuous:
                    target_mu, target_log_std = self.actor_net(feat_)
                    target_log_std = clip_nn_log_std(target_log_std)
                    target_pi, target_log_pi = squash_rsample(target_mu, target_log_std)
                else:
                    target_logits = self.actor_net(feat_)
                    target_cate_dist = tfp.distributions.Categorical(target_logits)
                    target_pi = target_cate_dist.sample()
                    target_log_pi = target_cate_dist.log_prob(target_pi)
                    target_pi = tf.one_hot(target_pi, self.a_counts, dtype=tf.float32)
                q1 = self.q1_net(feat, a)
                q1_target = self.q1_target_net(feat_, target_pi)
                q2 = self.q2_net(feat, a)
                q2_target = self.q2_target_net(feat_, target_pi)
                dc_r_q1 = tf.stop_gradient(r + self.gamma * (1 - done) * (q1_target - tf.exp(self.log_alpha) * target_log_pi))
                dc_r_q2 = tf.stop_gradient(r + self.gamma * (1 - done) * (q2_target - tf.exp(self.log_alpha) * target_log_pi))
                td_error1 = q1 - dc_r_q1
                td_error2 = q2 - dc_r_q2
                q1_loss = tf.reduce_mean(tf.square(td_error1) * isw)
                q2_loss = tf.reduce_mean(tf.square(td_error2) * isw)
                critic_loss = 0.5 * q1_loss + 0.5 * q2_loss + crsty_loss
            critic_grads = tape.gradient(critic_loss, self.critic_tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_tv)
            )

            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu, log_std = self.actor_net(feat)
                    log_std = clip_nn_log_std(log_std, self.log_std_min, self.log_std_max)
                    pi, log_pi = squash_rsample(mu, log_std)
                    entropy = gaussian_entropy(log_std)
                else:
                    logits = self.actor_net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    gumbel_noise = tf.cast(self.gumbel_dist.sample([batch_size, self.a_counts]), dtype=tf.float32)
                    _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_counts)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    pi = _pi_diff + _pi
                    log_pi = tf.reduce_sum(tf.multiply(logp_all, pi), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                q1_s_pi = self.q1_net(feat, pi)
                q2_s_pi = self.q2_net(feat, pi)
                actor_loss = -tf.reduce_mean(tf.minimum(q1_s_pi, q2_s_pi) - tf.exp(self.log_alpha) * log_pi)
            actor_grads = tape.gradient(actor_loss, self.actor_tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_tv)
            )

            if self.auto_adaption:
                with tf.GradientTape() as tape:
                    if self.is_continuous:
                        mu, log_std = self.actor_net(feat)
                        log_std = clip_nn_log_std(log_std, self.log_std_min, self.log_std_max)
                        norm_dist = tfp.distributions.Normal(loc=mu, scale=tf.exp(log_std))
                        log_pi = tf.reduce_sum(norm_dist.log_prob(norm_dist.sample()),axis=-1)
                    else:
                        logits = self.actor_net(feat)
                        cate_dist = tfp.distributions.Categorical(logits)
                        log_pi = cate_dist.log_prob(cate_dist.sample())
                    alpha_loss = -tf.reduce_mean(tf.exp(self.log_alpha) * tf.stop_gradient(log_pi - self.a_counts))
                alpha_grad = tape.gradient(alpha_loss, self.log_alpha)
                self.optimizer_alpha.apply_gradients(
                    [(alpha_grad, self.log_alpha)]
                )
            self.global_step.assign_add(1)
            summaries = dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/q1_loss', q1_loss],
                ['LOSS/q2_loss', q2_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/log_alpha', self.log_alpha],
                ['Statistics/alpha', tf.exp(self.log_alpha)],
                ['Statistics/entropy', entropy],
                ['Statistics/q_min', tf.reduce_min(tf.minimum(q1, q2))],
                ['Statistics/q_mean', tf.reduce_mean(tf.minimum(q1, q2))],
                ['Statistics/q_max', tf.reduce_max(tf.maximum(q1, q2))]
            ])
            if self.auto_adaption:
                summaries.update({
                    'LOSS/alpha_loss': alpha_loss
                })
            return (td_error1 + td_error2) / 2., summaries

    @tf.function(experimental_relax_shapes=True)
    def train_persistent(self, memories, isw, crsty_loss, cell_state):
        ss, vvss, a, r, done = memories
        batch_size = tf.shape(a)[0]
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                if self.is_continuous:
                    mu, log_std = self.actor_net(feat)
                    log_std = clip_nn_log_std(log_std, self.log_std_min, self.log_std_max)
                    pi, log_pi = squash_rsample(mu, log_std)
                    entropy = gaussian_entropy(log_std)
                    target_mu, target_log_std = self.actor_net(feat_)
                    target_log_std = clip_nn_log_std(target_log_std)
                    target_pi, target_log_pi = squash_rsample(target_mu, target_log_std)
                else:
                    logits = self.actor_net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    gumbel_noise = tf.cast(self.gumbel_dist.sample([batch_size, self.a_counts]), dtype=tf.float32)
                    _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_counts)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    pi = _pi_diff + _pi
                    log_pi = tf.reduce_sum(tf.multiply(logp_all, pi), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))

                    target_logits = self.actor_net(feat_)
                    target_cate_dist = tfp.distributions.Categorical(target_logits)
                    target_pi = target_cate_dist.sample()
                    target_log_pi = target_cate_dist.log_prob(target_pi)
                    target_pi = tf.one_hot(target_pi, self.a_counts, dtype=tf.float32)
                q1 = self.q1_net(feat, a)
                q1_target = self.q1_target_net(feat_, target_pi)
                q2 = self.q2_net(feat, a)
                q2_target = self.q2_target_net(feat_, target_pi)
                q1_s_pi = self.q1_net(feat, pi)
                q2_s_pi = self.q2_net(feat, pi)
                dc_r_q1 = tf.stop_gradient(r + self.gamma * (1 - done) * (q1_target - tf.exp(self.log_alpha) * target_log_pi))
                dc_r_q2 = tf.stop_gradient(r + self.gamma * (1 - done) * (q2_target - tf.exp(self.log_alpha) * target_log_pi))
                td_error1 = q1 - dc_r_q1
                td_error2 = q2 - dc_r_q2
                q1_loss = tf.reduce_mean(tf.square(td_error1) * isw)
                q2_loss = tf.reduce_mean(tf.square(td_error2) * isw)
                critic_loss = 0.5 * q1_loss + 0.5 * q2_loss + crsty_loss
                actor_loss = -tf.reduce_mean(tf.minimum(q1_s_pi, q2_s_pi) - tf.exp(self.log_alpha) * log_pi)
                if self.auto_adaption:
                    alpha_loss = -tf.reduce_mean(tf.exp(self.log_alpha) * tf.stop_gradient(log_pi - self.a_counts))
            critic_grads = tape.gradient(critic_loss, self.critic_tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_tv)
            )
            actor_grads = tape.gradient(actor_loss, self.actor_tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_tv)
            )
            if self.auto_adaption:
                alpha_grad = tape.gradient(alpha_loss, self.log_alpha)
                self.optimizer_alpha.apply_gradients(
                    [(alpha_grad, self.log_alpha)]
                )
            self.global_step.assign_add(1)
            summaries = dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/q1_loss', q1_loss],
                ['LOSS/q2_loss', q2_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/log_alpha', self.log_alpha],
                ['Statistics/alpha', tf.exp(self.log_alpha)],
                ['Statistics/entropy', entropy],
                ['Statistics/q_min', tf.reduce_min(tf.minimum(q1, q2))],
                ['Statistics/q_mean', tf.reduce_mean(tf.minimum(q1, q2))],
                ['Statistics/q_max', tf.reduce_max(tf.maximum(q1, q2))]
            ])
            if self.auto_adaption:
                summaries.update({
                    'LOSS/alpha_loss': alpha_loss
                })
            return (td_error1 + td_error2) / 2, summaries

    @tf.function(experimental_relax_shapes=True)
    def train_discrete(self, memories, isw, crsty_loss):
        ss, vvss, a, r, done = memories
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                q1_all = self.q1_net(feat)   # [B, A]
                q2_all = self.q2_net(feat)   # [B, A]
                q_function = lambda x: tf.reduce_sum(x*a, axis=-1, keepdims=True)   #[B, 1]
                q1 = q_function(q1_all)
                q2 = q_function(q2_all)
                target_logits = self.actor_net(feat_)   #[B, A]
                target_log_probs = tf.nn.log_softmax(target_logits)     #[B, A]
                q1_target = self.q1_target_net(feat_)   # [B, A]
                q2_target = self.q2_target_net(feat_)   # [B, A]
                v_target_function = lambda x: tf.reduce_sum(tf.exp(target_log_probs)*(x-tf.exp(self.log_alpha)*target_log_probs), axis=-1, keepdims=True)   #[B, 1]
                v1_target = v_target_function(q1_target)
                v2_target = v_target_function(q2_target)
                dc_r_q1 = tf.stop_gradient(r + self.gamma * (1 - done) * v1_target)
                dc_r_q2 = tf.stop_gradient(r + self.gamma * (1 - done) * v2_target)
                td_error1 = q1 - dc_r_q1
                td_error2 = q2 - dc_r_q2
                q1_loss = tf.reduce_mean(tf.square(td_error1) * isw)
                q2_loss = tf.reduce_mean(tf.square(td_error2) * isw)
                critic_loss = 0.5 * q1_loss + 0.5 * q2_loss + crsty_loss
            critic_grads = tape.gradient(critic_loss, self.critic_tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_tv)
            )

            with tf.GradientTape() as tape:
                logits = self.actor_net(feat)
                logp_all = tf.nn.log_softmax(logits)
                entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                q1_all = self.q1_net(feat)   # [B, A]
                q2_all = self.q2_net(feat)   # [B, A]
                q_all = tf.minimum(q1_all, q2_all)  # [B, A]
                actor_loss = -tf.reduce_mean(
                    tf.reduce_sum((q_all - tf.exp(self.log_alpha) * logp_all) * tf.exp(logp_all)) # [B, A] => [B,]
                )
            actor_grads = tape.gradient(actor_loss, self.actor_tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_tv)
            )

            if self.auto_adaption:
                with tf.GradientTape() as tape:
                    logits = self.actor_net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    corr = tf.stop_gradient(tf.reduce_sum((logp_all - self.a_counts) * tf.exp(logp_all), axis=-1))    #[B, A] => [B,]
                    alpha_loss = -tf.reduce_mean(tf.exp(self.log_alpha) * corr)
                alpha_grad = tape.gradient(alpha_loss, self.log_alpha)
                self.optimizer_alpha.apply_gradients(
                    [(alpha_grad, self.log_alpha)]
                )
            self.global_step.assign_add(1)
            summaries = dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/q1_loss', q1_loss],
                ['LOSS/q2_loss', q2_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/log_alpha', self.log_alpha],
                ['Statistics/alpha', tf.exp(self.log_alpha)],
                ['Statistics/entropy', entropy]
            ])
            if self.auto_adaption:
                summaries.update({
                    'LOSS/alpha_loss': alpha_loss
                })
            return (td_error1 + td_error2) / 2, summaries
