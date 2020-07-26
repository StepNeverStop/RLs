import rls
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from utils.tf2_utils import show_graph, get_TensorSpecs, gaussian_clip_rsample, gaussian_likelihood_sum, gaussian_entropy
from algos.tf2algos.base.on_policy import make_on_policy_class


class PPO(make_on_policy_class(mode='share')):
    '''
    Proximal Policy Optimization, https://arxiv.org/abs/1707.06347
    Emergence of Locomotion Behaviours in Rich Environments, http://arxiv.org/abs/1707.02286, DPPO
    '''

    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim,
                 is_continuous,

                 policy_epoch=4,
                 value_epoch=4,
                 beta=1.0e-3,
                 lr=5.0e-4,
                 lambda_=0.95,
                 epsilon=0.2,
                 value_epsilon=0.2,
                 share_net=True,
                 actor_lr=3e-4,
                 critic_lr=1e-3,
                 kl_reverse=False,
                 kl_target=0.02,
                 kl_target_cutoff=2,
                 kl_target_earlystop=4,
                 kl_beta=[0.7, 1.3],
                 kl_alpha=1.5,
                 kl_coef=1.0,
                 hidden_units={
                     'share': {
                         'continuous': {
                             'share': [32, 32],
                             'mu': [32, 32],
                             'v': [32, 32]
                         },
                         'discrete': {
                             'share': [32, 32],
                             'logits': [32, 32],
                             'v': [32, 32]
                         }
                     },
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'critic': [32, 32]
                 },
                 **kwargs):
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim=a_dim,
            is_continuous=is_continuous,
            **kwargs)
        self.beta = beta
        self.policy_epoch = policy_epoch
        self.value_epoch = value_epoch
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.value_epsilon = value_epsilon
        self.share_net = share_net
        self.kl_reverse = kl_reverse
        self.kl_target = kl_target
        self.kl_alpha = kl_alpha
        self.kl_coef = tf.constant(kl_coef, dtype=tf.float32)

        self.kl_cutoff = kl_target * kl_target_cutoff
        self.kl_stop = kl_target * kl_target_earlystop
        self.kl_low = kl_target * kl_beta[0]
        self.kl_high = kl_target * kl_beta[-1]

        if self.is_continuous:
            self.log_std = tf.Variable(initial_value=-0.5 * np.ones(self.a_dim, dtype=np.float32), trainable=True)
        if self.share_net:
            # self.TensorSpecs = get_TensorSpecs([self.s_dim], self.visual_dim, [self.a_dim], [1], [1], [1])
            if self.is_continuous:
                self.net = rls.a_c_v_continuous(self.feat_dim, self.a_dim, hidden_units['share']['continuous'])
                self.net_tv = self.net.trainable_variables + [self.log_std] + self.other_tv
            else:
                self.net = rls.a_c_v_discrete(self.feat_dim, self.a_dim, hidden_units['share']['discrete'])
                self.net_tv = self.net.trainable_variables + self.other_tv
            self.lr = self.init_lr(lr)
            self.optimizer = self.init_optimizer(self.lr)
            self.model_recorder(dict(
                model=self.net,
                optimizer=self.optimizer
            ))
        else:
            # self.actor_TensorSpecs = get_TensorSpecs([self.s_dim], self.visual_dim, [self.a_dim], [1], [1])
            # self.critic_TensorSpecs = get_TensorSpecs([self.s_dim], self.visual_dim, [1])
            if self.is_continuous:
                self.actor_net = rls.actor_mu(self.feat_dim, self.a_dim, hidden_units['actor_continuous'])
                self.actor_net_tv = self.actor_net.trainable_variables + [self.log_std]
            else:
                self.actor_net = rls.actor_discrete(self.feat_dim, self.a_dim, hidden_units['actor_discrete'])
                self.actor_net_tv = self.actor_net.trainable_variables
            self.critic_net = rls.critic_v(self.feat_dim, hidden_units['critic'])
            self.critic_tv = self.critic_net.trainable_variables + self.other_tv
            self.actor_lr, self.critic_lr = map(self.init_lr, [actor_lr, critic_lr])
            self.optimizer_actor, self.optimizer_critic = map(self.init_optimizer, [self.actor_lr, self.critic_lr])
            self.model_recorder(dict(
                actor=self.actor_net,
                critic=self.critic_net,
                optimizer_actor=self.optimizer_actor,
                optimizer_critic=self.optimizer_critic
            ))

        self.initialize_data_buffer(
            data_name_list=['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done', 'value', 'log_prob'])

    def show_logo(self):
        self.logger.info('''
　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘ　　　　　
　　　　　ｘｘ　　ｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　　　　　ｘｘｘ　ｘｘｘ　　　　
　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　ｘｘ　　　ｘｘ　　　　
　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　ｘｘ　　　ｘｘｘ　　　
　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　　ｘｘｘ　　　
　　　　　ｘ　　　　　　　　　　　　　　ｘ　　　　　　　　　　　　　ｘｘ　　　ｘｘｘ　　　
　　　　　ｘ　　　　　　　　　　　　　　ｘ　　　　　　　　　　　　　ｘｘ　　　ｘｘ　　　　
　　　　　ｘ　　　　　　　　　　　　　　ｘ　　　　　　　　　　　　　ｘｘ　　ｘｘｘ　　　　
　　　ｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　　ｘｘｘｘｘ　　
        ''')

    def choose_action(self, s, visual_s, evaluation=False):
        a, value, log_prob, self.cell_state = self._get_action(s, visual_s, self.cell_state)
        a = a.numpy()
        self._value = np.squeeze(value.numpy())
        self._log_prob = np.squeeze(log_prob.numpy()) + 1e-10
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state):
        with tf.device(self.device):
            feat, cell_state = self.get_feature(s, visual_s, cell_state=cell_state, record_cs=True)
            if self.is_continuous:
                if self.share_net:
                    mu, value = self.net(feat)
                else:
                    mu = self.actor_net(feat)
                    value = self.critic_net(feat)
                sample_op, _ = gaussian_clip_rsample(mu, self.log_std)
                log_prob = gaussian_likelihood_sum(sample_op, mu, self.log_std)
            else:
                if self.share_net:
                    logits, value = self.net(feat)
                else:
                    logits = self.actor_net(feat)
                    value = self.critic_net(feat)
                norm_dist = tfp.distributions.Categorical(logits)
                sample_op = norm_dist.sample()
                log_prob = norm_dist.log_prob(sample_op)
        return sample_op, value, log_prob, cell_state

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "store_data need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store_data need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store_data need done type is np.ndarray"
        self._running_average(s)
        self.data.add(s, visual_s, a, r, s_, visual_s_, done, self._value, self._log_prob)

    @tf.function
    def _get_value(self, feat):
        with tf.device(self.device):
            if self.share_net:
                _, value = self.net(feat)
            else:
                value = self.critic_net(feat)
            return value

    def calculate_statistics(self):
        feat, self.cell_state = self.get_feature(self.data.last_s(), self.data.last_visual_s(), cell_state=self.cell_state, record_cs=True)
        init_value = np.squeeze(self._get_value(feat).numpy())
        self.data.cal_dc_r(self.gamma, init_value)
        self.data.cal_td_error(self.gamma, init_value)
        self.data.cal_gae_adv(self.lambda_, self.gamma)

    # @show_graph(name='ppo_net')
    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        def _train(data, crsty_loss, cell_state):
            early_step = 0
            if self.share_net:
                for i in range(self.policy_epoch):
                    actor_loss, critic_loss, entropy, kl = self.train_share(
                        data,
                        self.kl_coef,
                        crsty_loss,
                        cell_state
                    )
                    if kl > self.kl_stop:
                        early_step = i
                        break
            else:
                for i in range(self.policy_epoch):
                    s, visual_s, a, dc_r, old_log_prob, advantage, old_value = data
                    actor_loss, entropy, kl = self.train_actor(
                        (s, visual_s, a, old_log_prob, advantage),
                        self.kl_coef,
                        cell_state
                    )
                    if kl > self.kl_stop:
                        early_step = i
                        break

                for _ in range(self.value_epoch):
                    critic_loss = self.train_critic(
                        (s, visual_s, dc_r, old_value),
                        crsty_loss,
                        cell_state
                    )

            # https://github.com/joschu/modular_rl/blob/6970cde3da265cf2a98537250fea5e0c0d9a7639/modular_rl/ppo.py#L93
            if kl > self.kl_high:
                self.kl_coef *= self.kl_alpha
            elif kl < self.kl_low:
                self.kl_coef /= self.kl_alpha

            summaries = dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/kl', kl],
                ['Statistics/kl_coef', self.kl_coef],
                ['Statistics/early_step', early_step],
                ['Statistics/entropy', entropy]
            ])
            return summaries

        if self.share_net:
            summary_dict = dict([['LEARNING_RATE/lr', self.lr(self.train_step)]])
        else:
            summary_dict = dict([
                ['LEARNING_RATE/actor_lr', self.actor_lr(self.train_step)],
                ['LEARNING_RATE/critic_lr', self.critic_lr(self.train_step)]
            ])

        self._learn(function_dict={
            'calculate_statistics': self.calculate_statistics,
            'train_function': _train,
            'train_data_list': ['s', 'visual_s', 'a', 'discounted_reward', 'log_prob', 'gae_adv', 'value'],
            'summary_dict': summary_dict
        })

    @tf.function(experimental_relax_shapes=True)
    def train_share(self, memories, kl_coef, crsty_loss, cell_state):
        s, visual_s, a, dc_r, old_log_prob, advantage, old_value = memories
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat = self.get_feature(s, visual_s, cell_state=cell_state)
                if self.is_continuous:
                    mu, value = self.net(feat)
                    new_log_prob = gaussian_likelihood_sum(a, mu, self.log_std)
                    entropy = gaussian_entropy(self.log_std)
                else:
                    logits, value = self.net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    new_log_prob = tf.reduce_sum(a * logp_all, axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                ratio = tf.exp(new_log_prob - old_log_prob)

                # https://github.com/joschu/modular_rl/blob/6970cde3da265cf2a98537250fea5e0c0d9a7639/modular_rl/ppo.py#L40
                if self.kl_reverse:
                    kl = tf.reduce_mean(new_log_prob - old_log_prob)
                else:
                    kl = tf.reduce_mean(old_log_prob - new_log_prob)    # a sample estimate for KL-divergence, easy to compute
                surrogate = ratio * advantage

                # https://github.com/llSourcell/OpenAI_Five_vs_Dota2_Explained/blob/c5def7e57aa70785c2394ea2eeb3e5f66ad59a53/train.py#L154
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
                actor_loss = pi_loss + kl_loss + extra_loss
                value_loss = 0.5 * tf.reduce_mean(td_square)
                loss = actor_loss + 1.0 * value_loss - self.beta * entropy + crsty_loss
            loss_grads = tape.gradient(loss, self.net_tv)
            self.optimizer.apply_gradients(
                zip(loss_grads, self.net_tv)
            )
            self.global_step.assign_add(1)
            return actor_loss, value_loss, entropy, kl

    @tf.function(experimental_relax_shapes=True)
    def train_actor(self, memories, kl_coef, cell_state):
        s, visual_s, a, old_log_prob, advantage = memories
        with tf.device(self.device):
            feat = self.get_feature(s, visual_s, cell_state=cell_state)
            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu = self.actor_net(feat)
                    new_log_prob = gaussian_likelihood_sum(a, mu, self.log_std)
                    entropy = gaussian_entropy(self.log_std)
                else:
                    logits = self.actor_net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    new_log_prob = tf.reduce_sum(a * logp_all, axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                ratio = tf.exp(new_log_prob - old_log_prob)
                kl = tf.reduce_mean(old_log_prob - new_log_prob)
                surrogate = ratio * advantage
                min_adv = tf.where(advantage > 0, (1 + self.epsilon) * advantage, (1 - self.epsilon) * advantage)
                pi_loss = -(tf.reduce_mean(tf.minimum(surrogate, min_adv)) + self.beta * entropy)

                kl_loss = kl_coef * kl
                extra_loss = 1000.0 * tf.square(tf.maximum(0., kl - self.kl_cutoff))
                actor_loss = pi_loss + kl_loss + extra_loss

            actor_grads = tape.gradient(actor_loss, self.actor_net_tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net_tv)
            )
            self.global_step.assign_add(1)
            return actor_loss, entropy, kl

    @tf.function(experimental_relax_shapes=True)
    def train_critic(self, memories, crsty_loss, cell_state):
        s, visual_s, dc_r, old_value = memories
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat = self.get_feature(s, visual_s, cell_state=cell_state)
                value = self.critic_net(feat)

                value_clip = old_value + tf.clip_by_value(value - old_value, -self.value_epsilon, self.value_epsilon)
                td_error = dc_r - value
                td_error_clip = dc_r - value_clip
                td_square = tf.maximum(tf.square(td_error), tf.square(td_error_clip))

                value_loss = 0.5 * tf.reduce_mean(td_square) + crsty_loss
            critic_grads = tape.gradient(value_loss, self.critic_tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_tv)
            )
            return value_loss
