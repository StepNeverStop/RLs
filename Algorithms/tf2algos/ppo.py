import Nn
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from utils.tf2_utils import show_graph, get_TensorSpecs, gaussian_clip_rsample, gaussian_likelihood_sum, gaussian_entropy
from Algorithms.tf2algos.base.on_policy import On_Policy


class PPO(On_Policy):
    '''
    Proximal Policy Optimization, https://arxiv.org/abs/1707.06347
    '''
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,

                 epoch=5,
                 beta=1.0e-3,
                 lr=5.0e-4,
                 lambda_=0.95,
                 epsilon=0.2,
                 share_net=True,
                 actor_lr=3e-4,
                 critic_lr=1e-3,
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
            a_dim_or_list=a_dim_or_list,
            is_continuous=is_continuous,
            **kwargs)
        self.beta = beta
        self.epoch = epoch
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.share_net = share_net

        if self.is_continuous:
            self.log_std = tf.Variable(initial_value=-0.5 * np.ones(self.a_counts, dtype=np.float32), trainable=True)
        if self.share_net:
            # self.TensorSpecs = get_TensorSpecs([self.s_dim], self.visual_dim, [self.a_counts], [1], [1], [1])
            if self.is_continuous:
                self.net = Nn.a_c_v_continuous(self.rnn_net.hdim, self.a_counts, hidden_units['share']['continuous'])
                self.net_tv = self.net.trainable_variables + [self.log_std] + self.other_tv
            else:
                self.net = Nn.a_c_v_discrete(self.rnn_net.hdim, self.a_counts, hidden_units['share']['discrete'])
                self.net_tv = self.net.trainable_variables + self.other_tv
            self.lr = tf.keras.optimizers.schedules.PolynomialDecay(lr, self.max_episode, 1e-10, power=1.0)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr(self.episode))
            self.model_recorder(dict(
                model=self.net,
                optimizer=self.optimizer
                ))
        else:
            # self.actor_TensorSpecs = get_TensorSpecs([self.s_dim], self.visual_dim, [self.a_counts], [1], [1])
            # self.critic_TensorSpecs = get_TensorSpecs([self.s_dim], self.visual_dim, [1])
            if self.is_continuous:
                self.actor_net = Nn.actor_mu(self.rnn_net.hdim, self.a_counts, hidden_units['actor_continuous'])
                self.actor_net_tv = self.actor_net.trainable_variables+ [self.log_std]
            else:
                self.actor_net = Nn.actor_discrete(self.rnn_net.hdim, self.a_counts, hidden_units['actor_discrete'])
                self.actor_net_tv = self.actor_net.trainable_variables
            self.critic_net = Nn.critic_v(self.rnn_net.hdim, hidden_units['critic'])
            self.critic_tv = self.critic_net.trainable_variables + self.other_tv
            self.actor_lr = tf.keras.optimizers.schedules.PolynomialDecay(actor_lr, self.max_episode, 1e-10, power=1.0)
            self.critic_lr = tf.keras.optimizers.schedules.PolynomialDecay(critic_lr, self.max_episode, 1e-10, power=1.0)
            self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.actor_lr(self.episode))
            self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.critic_lr(self.episode))
            self.model_recorder(dict(
                actor=self.actor_net,
                critic=self.critic_net,
                optimizer_actor=self.optimizer_actor,
                optimizer_critic=self.optimizer_critic
                ))
            
        self.initialize_data_buffer(
            data_name_list=['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done', 'value', 'log_prob'])

    def show_logo(self):
        self.recorder.logger.info('''
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
            feat, cell_state = self.get_feature(s, visual_s, cell_state=cell_state, record_cs=True, train=False)
            if self.is_continuous:
                if self.share_net:
                    mu, value = self.net(feat)
                else:
                    mu = self.actor_net(feat)
                    value = self.critic_net(feat)
                sample_op, _ = gaussian_clip_rsample(mu, self.log_std)
                log_prob = gaussian_likelihood_sum(mu, sample_op, self.log_std)
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
        feat, self.cell_state = self.get_feature(self.data.last_s(), self.data.last_visual_s(), cell_state=self.cell_state, record_cs=True, train=False)
        init_value = np.squeeze(self._get_value(feat).numpy())
        self.data.cal_dc_r(self.gamma, init_value)
        self.data.cal_td_error(self.gamma, init_value)
        self.data.cal_gae_adv(self.lambda_, self.gamma)

    # @show_graph(name='ppo_net')
    def learn(self, **kwargs):
        self.episode = kwargs['episode']

        def _train(data, crsty_loss, cell_state):
            if self.share_net:
                    actor_loss, critic_loss, entropy, kl = self.train_share(
                        data,
                        crsty_loss,
                        cell_state
                        )
            else:
                s, visual_s, a, dc_r, old_log_prob, advantage = data
                actor_loss, entropy, kl = self.train_actor(
                    (s, visual_s, a, old_log_prob, advantage),
                    cell_state
                )
                # if kl > 1.5 * 0.01:
                #     break
                critic_loss = self.train_critic(
                    (s, visual_s, dc_r),
                    crsty_loss,
                    cell_state
                )

            summaries = dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/critic_loss', critic_loss],
                ['Statistics/entropy', entropy]
            ])
            return summaries

        if self.share_net:
            summary_dict = dict([['LEARNING_RATE/lr', self.lr(self.episode)]])
        else:
            summary_dict =dict([
                ['LEARNING_RATE/actor_lr', self.actor_lr(self.episode)],
                ['LEARNING_RATE/critic_lr', self.critic_lr(self.episode)]
            ])

        self._learn(epoch=self.epoch,
                    function_dict={
                        'calculate_statistics': self.calculate_statistics,
                        'train_function': _train,
                        'train_data_list': ['s', 'visual_s', 'a', 'discounted_reward', 'log_prob', 'gae_adv'],
                        'summary_dict': summary_dict
                    })

    @tf.function(experimental_relax_shapes=True)
    def train_share(self, memories, crsty_loss, cell_state):
        s, visual_s, a, dc_r, old_log_prob, advantage = memories
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat = self.get_feature(s, visual_s, cell_state=cell_state)
                if self.is_continuous:
                    mu, value = self.net(feat)
                    new_log_prob = gaussian_likelihood_sum(mu, a, self.log_std)
                    entropy = gaussian_entropy(self.log_std)
                else:
                    logits, value = self.net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    new_log_prob = tf.reduce_sum(a * logp_all, axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                ratio = tf.exp(new_log_prob - old_log_prob)
                kl = tf.reduce_mean(old_log_prob - new_log_prob)
                surrogate = ratio * advantage
                td_error = dc_r - value
                actor_loss = tf.reduce_mean(
                    tf.minimum(
                        surrogate,
                        tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
                    ))
                value_loss = tf.reduce_mean(tf.square(td_error))
                loss = -(actor_loss - 1.0 * value_loss + self.beta * entropy) + crsty_loss
            loss_grads = tape.gradient(loss, self.net_tv)
            self.optimizer.apply_gradients(
                zip(loss_grads, self.net_tv)
            )
            self.global_step.assign_add(1)
            return actor_loss, value_loss, entropy, kl

    @tf.function(experimental_relax_shapes=True)
    def train_actor(self, memories, cell_state):
        s, visual_s, a, old_log_prob, advantage = memories
        with tf.device(self.device):
            feat = self.get_feature(s, visual_s, cell_state=cell_state)
            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu = self.actor_net(feat)
                    new_log_prob = gaussian_likelihood_sum(mu, a, self.log_std)
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
                actor_loss = -(tf.reduce_mean(tf.minimum(surrogate, min_adv)) + self.beta * entropy)
            actor_grads = tape.gradient(actor_loss, self.actor_net_tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_net_tv)
            )
            self.global_step.assign_add(1)
            return actor_loss, entropy, kl

    @tf.function(experimental_relax_shapes=True)
    def train_critic(self, memories, crsty_loss, cell_state):
        s, visual_s, dc_r = memories
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat = self.get_feature(s, visual_s, cell_state=cell_state)
                value = self.critic_net(feat)
                td_error = dc_r - value
                value_loss = tf.reduce_mean(tf.square(td_error)) + crsty_loss
            critic_grads = tape.gradient(value_loss, self.critic_tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_tv)
            )
            return value_loss
