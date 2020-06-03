import Nn
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from utils.tf2_utils import gaussian_clip_rsample, gaussian_likelihood_sum, gaussian_entropy
from Algorithms.tf2algos.base.on_policy import make_on_policy_class


class PPOC(make_on_policy_class(mode='share')):
    '''
    Learnings Options End-to-End for Continuous Action Tasks, PPOC, http://arxiv.org/abs/1712.00004
    '''
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim,
                 is_continuous,

                 options_num=4,
                 dc=0.01,
                 terminal_mask=False,
                 o_beta=1.0e-3,
                 epoch=4,
                 pi_beta=1.0e-3,
                 lr=5.0e-4,
                 lambda_=0.95,
                 epsilon=0.2,
                 value_epsilon=0.2,
                 kl_reverse=False,
                 kl_target=0.02,
                 kl_target_cutoff=2,
                 kl_target_earlystop=4,
                 kl_beta=[0.7, 1.3],
                 kl_alpha=1.5,
                 kl_coef=1.0,
                 hidden_units={
                     'share': [32, 32],
                     'q': [32, 32],
                     'intra_option': [32, 32],
                     'termination': [32, 32],
                     'o': [32, 32]
                 },
                 **kwargs):
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim=a_dim,
            is_continuous=is_continuous,
            **kwargs)
        self.pi_beta = pi_beta
        self.epoch = epoch
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.value_epsilon = value_epsilon
        self.kl_reverse = kl_reverse
        self.kl_target = kl_target
        self.kl_alpha = kl_alpha
        self.kl_coef = tf.constant(kl_coef, dtype=tf.float32)

        self.kl_cutoff = kl_target * kl_target_cutoff
        self.kl_stop = kl_target * kl_target_earlystop
        self.kl_low = kl_target * kl_beta[0]
        self.kl_high = kl_target * kl_beta[-1]

        self.options_num = options_num
        self.dc = dc
        self.terminal_mask = terminal_mask
        self.o_beta = o_beta


        self.net = Nn.ppoc_share(self.rnn_net.hdim, self.a_dim, self.options_num, hidden_units, self.is_continuous)
        if self.is_continuous:
            self.log_std = tf.Variable(initial_value=-0.5 * np.ones((self.options_num, self.a_dim), dtype=np.float32), trainable=True)   # [P, A]
            self.net_tv = self.net.trainable_variables + [self.log_std] + self.other_tv
        else:
            self.net_tv = self.net.trainable_variables + self.other_tv
        self.lr = self.init_lr(lr)
        self.optimizer = self.init_optimizer(self.lr)
        self.model_recorder(dict(
            model=self.net,
            optimizer=self.optimizer
            ))
            
        self.initialize_data_buffer(
            data_name_list=['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done', 'value', 'log_prob', 'o_log_prob', 'beta_adv', 'last_options', 'options'])

    def show_logo(self):
        self.recorder.logger.info('''
　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘ　　　　
　　　　　ｘｘ　　ｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　　　　　ｘｘｘ　ｘｘｘ　　　　　　　　ｘｘｘ　　ｘｘ　　　　
　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘ　　　　ｘｘ　　　
　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　ｘｘ　　　ｘｘｘ　　　　　　　ｘｘ　　　　　　　　　
　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　　ｘｘｘ　　　　　　ｘｘｘ　　　　　　　　　
　　　　　ｘ　　　　　　　　　　　　　　ｘ　　　　　　　　　　　　　ｘｘ　　　ｘｘｘ　　　　　　ｘｘｘ　　　　　　　　　
　　　　　ｘ　　　　　　　　　　　　　　ｘ　　　　　　　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘ　　　　ｘｘ　　　
　　　　　ｘ　　　　　　　　　　　　　　ｘ　　　　　　　　　　　　　ｘｘ　　ｘｘｘ　　　　　　　　ｘｘｘ　　ｘｘｘ　　　
　　　ｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘ　　
        ''')

    def reset(self):
        super().reset()
        self._done_mask = np.full(self.n_agents, True)

    def partial_reset(self, done):
        super().partial_reset(done)
        self._done_mask = done
        
    def _generate_random_options(self):
        return tf.constant(np.random.randint(0, self.options_num, self.n_agents), dtype=tf.int32)

    def choose_action(self, s, visual_s, evaluation=False):
        if not hasattr(self, 'options'):
            self.options = self._generate_random_options()
        self.last_options = self.options
        if not hasattr(self, 'oc_mask'):
            self.oc_mask = tf.constant(np.zeros(self.n_agents), dtype=tf.int32)

        a, value, log_prob, o_log_prob, beta_adv, new_options, self.cell_state = self._get_action(s, visual_s, self.cell_state, self.options)
        a = a.numpy()
        new_options = tf.where(self._done_mask, max_options, new_options)
        self._done_mask = np.full(self.n_agents, False)
        self._value = np.squeeze(value.numpy())
        self._log_prob = np.squeeze(log_prob.numpy()) + 1e-10
        self._o_log_prob = np.squeeze(o_log_prob.numpy()) + 1e-10
        self._beta_adv = np.squeeze(beta_adv.numpy()) + self.dc
        self.oc_mask = (new_options == self.options).numpy()  # equal means no change
        self.options = new_options
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state, options):
        with tf.device(self.device):
            feat, cell_state = self.get_feature(s, visual_s, cell_state=cell_state, record_cs=True, train=False)
            q, pi, beta, o = self.net(feat) # [B, P], [B, P, A], [B, P], [B, P]
            options_onehot = tf.one_hot(options, self.options_num, dtype=tf.float32)    # [B, P]
            options_onehot_expanded = tf.expand_dims(options_onehot, axis=-1)  # [B, P, 1]
            pi = tf.reduce_sum(pi * options_onehot_expanded, axis=1) # [B, A]
            if self.is_continuous:
                log_std = tf.gather(self.log_std, options)
                mu = pi
                sample_op, _ = gaussian_clip_rsample(mu, log_std)
                log_prob = gaussian_likelihood_sum(sample_op, mu, log_std)
            else:
                logits = pi
                norm_dist = tfp.distributions.Categorical(logits)
                sample_op = norm_dist.sample()
                log_prob = norm_dist.log_prob(sample_op)
            o_log_prob = tf.reduce_sum(o*options_onehot, axis=-1)   # [B, ]
            q_o = tf.reduce_sum(q*options_onehot, axis=-1)  # [B, ]
            beta_adv = q_o - tf.reduce_sum(q*tf.math.exp(o), axis=-1)   # [B, ]
            option_norm_dist = tfp.distributions.Categorical(probs=tf.math.exp(o)) 
            sample_options = option_norm_dist.sample()
            beta_probs = tf.reduce_sum(beta * options_onehot, axis=1)   # [B, P] => [B,]
            beta_dist = tfp.distributions.Bernoulli(probs=beta_probs)
            new_options = tf.where(beta_dist.sample()<1, options, sample_options)    # <1 则不改变op， =1 则改变op
        return sample_op, q_o, log_prob, o_log_prob, beta_adv, new_options, cell_state

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "store_data need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store_data need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store_data need done type is np.ndarray"
        r -= (1 - self.oc_mask) * self.dc
        self.data.add(s, visual_s, a, r, s_, visual_s_, done, self._value, self._log_prob, self._o_log_prob, self._beta_adv, self.last_options, self.options)
        self.oc_mask = tf.zeros_like(self.oc_mask)

    @tf.function
    def _get_value(self, feat, options):
        options = tf.cast(options, tf.int32)
        with tf.device(self.device):
            options_onehot = tf.one_hot(options, self.options_num, dtype=tf.float32)    # [B, P]
            q, _, _, _ = self.net(feat)
            q_o = tf.reduce_sum(q*options_onehot, axis=-1)  # [B, ]
            return q_o

    def calculate_statistics(self):
        feat, self.cell_state = self.get_feature(self.data.last_s(), self.data.last_visual_s(), cell_state=self.cell_state, record_cs=True, train=False)
        init_value = np.squeeze(self._get_value(feat, self.data.buffer['options'][-1]).numpy())
        self.data.cal_dc_r(self.gamma, init_value)
        self.data.cal_td_error(self.gamma, init_value)
        self.data.cal_gae_adv(self.lambda_, self.gamma)

    def learn(self, **kwargs):
        self.episode = kwargs['episode']

        def _train(data, crsty_loss, cell_state):
            early_step = 0
            for i in range(self.epoch):
                loss, pi_loss, q_loss, o_loss, beta_loss, entropy, o_entropy, kl = self.train_share(
                    data,
                    self.kl_coef,
                    crsty_loss,
                    cell_state
                    )
                if kl > self.kl_stop:
                    early_step = i
                    break

            if kl > self.kl_high:
                self.kl_coef *= self.kl_alpha
            elif kl < self.kl_low:
                self.kl_coef /= self.kl_alpha

            summaries = dict([
                ['LOSS/loss', loss],
                ['LOSS/loss', pi_loss],
                ['LOSS/loss', q_loss],
                ['LOSS/loss', o_loss],
                ['LOSS/loss', beta_loss],
                ['Statistics/kl', kl],
                ['Statistics/entropy', entropy],
                ['Statistics/o_entropy', o_entropy],
                ['Statistics/kl_coef', self.kl_coef],
                ['Statistics/early_step', early_step],
            ])
            return summaries

        summary_dict = dict([['LEARNING_RATE/lr', self.lr(self.episode)]])

        self._learn(function_dict={
                        'calculate_statistics': self.calculate_statistics,
                        'train_function': _train,
                        'train_data_list': ['s', 'visual_s', 'a', 'discounted_reward', 'log_prob', 'gae_adv', 'value', 'o_log_prob', 'beta_adv', 'last_options', 'options'],
                        'summary_dict': summary_dict
                    })

    @tf.function(experimental_relax_shapes=True)
    def train_share(self, memories, kl_coef, crsty_loss, cell_state):
        s, visual_s, a, dc_r, old_log_prob, advantage, old_value, old_o_log_prob, beta_advantage, last_options, options = memories
        last_options = tf.reshape(tf.cast(last_options, tf.int32), (-1,)) # [B, 1] => [B,]
        options = tf.reshape(tf.cast(options, tf.int32), (-1,))
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat = self.get_feature(s, visual_s, cell_state=cell_state)
                q, pi, beta, o = self.net(feat) # [B, P], [B, P, A], [B, P], [B, P]

                options_onehot = tf.one_hot(options, self.options_num, dtype=tf.float32)    # [B, P]
                options_onehot_expanded = tf.expand_dims(options_onehot, axis=-1)  # [B, P, 1]
                last_options_onehot = tf.one_hot(last_options, self.options_num, dtype=tf.float32)    # [B,] => [B, P]

                pi = tf.reduce_sum(pi * options_onehot_expanded, axis=1) # [B, P, A] => [B, A]
                value = tf.reduce_sum(q * options_onehot, axis=1, keepdims=True)    # [B, 1]

                if self.is_continuous:
                    log_std = tf.gather(self.log_std, options)
                    mu = pi # [B, A]
                    new_log_prob = gaussian_likelihood_sum(a, mu, log_std)
                    entropy = gaussian_entropy(log_std)
                else:
                    logits = pi #[B, A]
                    logp_all = tf.nn.log_softmax(logits)
                    new_log_prob = tf.reduce_sum(a * logp_all, axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                ratio = tf.exp(new_log_prob - old_log_prob)

                if self.kl_reverse:
                    kl = tf.reduce_mean(new_log_prob - old_log_prob)
                else:
                    kl = tf.reduce_mean(old_log_prob - new_log_prob)    # a sample estimate for KL-divergence, easy to compute
                surrogate = ratio * advantage
                
                value_clip = old_value + tf.clip_by_value(value - old_value, -self.value_epsilon, self.value_epsilon)
                td_error = dc_r - value
                td_error_clip = dc_r - value_clip
                td_square = tf.maximum(tf.square(td_error), tf.square(td_error_clip))

                pi_loss = -tf.reduce_mean(
                    tf.minimum(
                        surrogate,
                        tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
                    ))
                kl_loss = kl_coef * kl
                extra_loss = 1000.0 * tf.square(tf.maximum(0., kl - self.kl_cutoff))
                pi_loss = pi_loss + kl_loss + extra_loss
                q_loss = 0.5 * tf.reduce_mean(td_square)

                beta_s = tf.reduce_sum(beta * last_options_onehot, axis=-1, keepdims=True)   # [B, 1]
                beta_loss = tf.reduce_mean(beta_s * beta_advantage)
                if self.terminal_mask:
                    beta_loss *= (1 - done)

                o_log_prob = tf.reduce_sum(o*options_onehot, axis=-1, keepdims=True)   # [B, 1]
                o_ratio = tf.exp(o_log_prob - old_o_log_prob)
                o_entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(o) * o, axis=1, keepdims=True))
                o_loss = -tf.reduce_mean(
                    tf.minimum(
                        o_ratio * advantage,
                        tf.clip_by_value(o_ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
                    ))

                loss = pi_loss + 1.0 * q_loss + o_loss + beta_loss - self.pi_beta * entropy - self.o_beta * o_entropy + crsty_loss
            loss_grads = tape.gradient(loss, self.net_tv)
            self.optimizer.apply_gradients(
                zip(loss_grads, self.net_tv)
            )
            self.global_step.assign_add(1)
            return loss, pi_loss, q_loss, o_loss, beta_loss, entropy, o_entropy, kl

