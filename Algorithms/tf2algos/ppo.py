import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import Nn
from utils.sth import sth
from utils.np_utils import normalization, standardization
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
            self.TensorSpecs = get_TensorSpecs([self.s_dim], self.visual_dim, [self.a_counts], [1], [1], [1])
            self.visual_net = Nn.VisualNet('visual_net', self.visual_dim)
            if self.is_continuous:
                self.net = Nn.a_c_v_continuous(self.s_dim, self.a_counts, 'ppo_net', hidden_units['share']['continuous'], visual_net=self.visual_net)
                self.net.tv += [self.log_std]
            else:
                self.net = Nn.a_c_v_discrete(self.s_dim, self.a_counts, 'ppo_net', hidden_units['share']['discrete'], visual_net=self.visual_net)
            self.lr = tf.keras.optimizers.schedules.PolynomialDecay(lr, self.max_episode, 1e-10, power=1.0)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr(self.episode))
        else:
            self.actor_TensorSpecs = get_TensorSpecs([self.s_dim], self.visual_dim, [self.a_counts], [1], [1])
            self.critic_TensorSpecs = get_TensorSpecs([self.s_dim], self.visual_dim, [1])
            self.actor_visual_net = Nn.VisualNet('actor_visual_net', self.visual_dim)
            self.critic_visual_net = Nn.VisualNet('critic_visual_net', self.visual_dim)
            if self.is_continuous:
                self.actor_net = Nn.actor_mu(self.s_dim, self.a_counts, 'actor_net', hidden_units['actor_continuous'], visual_net=self.actor_visual_net)
                self.actor_net.tv += [self.log_std]
            else:
                self.actor_net = Nn.actor_discrete(self.s_dim, self.a_counts, 'actor_net', hidden_units['actor_discrete'], visual_net=self.actor_visual_net)
            self.critic_net = Nn.critic_v(self.s_dim, 'critic_net', hidden_units['critic'], visual_net=self.critic_visual_net)
            self.actor_lr = tf.keras.optimizers.schedules.PolynomialDecay(actor_lr, self.max_episode, 1e-10, power=1.0)
            self.critic_lr = tf.keras.optimizers.schedules.PolynomialDecay(critic_lr, self.max_episode, 1e-10, power=1.0)
            self.optimizer_actor = tf.keras.optimizers.Adam(learning_rate=self.actor_lr(self.episode))
            self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.critic_lr(self.episode))

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
        a = self._get_action(s, visual_s, evaluation).numpy()
        return a if self.is_continuous else sth.int2action_index(a, self.a_dim_or_list)

    @tf.function
    def _get_action(self, s, visual_s, evaluation):
        s, visual_s = self.cast(s, visual_s)
        with tf.device(self.device):
            if self.is_continuous:
                if self.share_net:
                    mu, _ = self.net(s, visual_s)
                else:
                    mu = self.actor_net(s, visual_s)
                sample_op, _ = gaussian_clip_rsample(mu, self.log_std)
            else:
                if self.share_net:
                    logits, _ = self.net(s, visual_s)
                else:
                    logits = self.actor_net(s, visual_s)
                norm_dist = tfp.distributions.Categorical(logits)
                sample_op = norm_dist.sample()
        return sample_op

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "store_data need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store_data need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store_data need done type is np.ndarray"
        if not self.is_continuous:
            a = sth.action_index2one_hot(a, self.a_dim_or_list)
        self.data = self.data.append({
            's': s,
            'visual_s': visual_s,
            'a': a,
            'r': r,
            'done': done,
            'value': np.squeeze(self._get_value(s, visual_s).numpy()),
            'log_prob': self._get_log_prob(s, visual_s, a).numpy() + 1e-10
        }, ignore_index=True)
        self.s_ = s_
        self.visual_s_ = visual_s_

    @tf.function
    def _get_value(self, s, visual_s):
        s, visual_s = self.cast(s, visual_s)
        with tf.device(self.device):
            if self.share_net:
                _, value = self.net(s, visual_s)
            else:
                value = self.critic_net(s, visual_s)
            return value

    @tf.function
    def _get_log_prob(self, s, visual_s, a):
        s, visual_s, a = self.cast(s, visual_s, a)
        with tf.device(self.device):
            if self.is_continuous:
                if self.share_net:
                    mu, _ = self.net(s, visual_s)
                else:
                    mu = self.actor_net(s, visual_s)
                new_log_prob = gaussian_likelihood_sum(mu, a, self.log_std)
            else:
                if self.share_net:
                    logits, _ = self.net(s, visual_s)
                else:
                    logits = self.actor_net(s, visual_s)
                logp_all = tf.nn.log_softmax(logits)
                new_log_prob = tf.reduce_sum(a * logp_all, axis=1, keepdims=True)
            return new_log_prob

    def calculate_statistics(self):
        init_value = np.squeeze(self._get_value(self.s_, self.visual_s_).numpy())
        self.data['total_reward'] = sth.discounted_sum(self.data.r.values, 1, init_value, self.data.done.values)
        self.data['discounted_reward'] = sth.discounted_sum(self.data.r.values, self.gamma, init_value, self.data.done.values)
        self.data['td_error'] = sth.discounted_sum_minus(
            self.data.r.values,
            self.gamma,
            init_value,
            self.data.done.values,
            self.data.value.values
        )
        # GAE
        adv = np.asarray(sth.discounted_sum(
            self.data.td_error.values,
            self.lambda_ * self.gamma,
            0,
            self.data.done.values
        ))
        self.data['advantage'] = list(standardization(adv))
        # self.data.to_excel(self.recorder.excel_writer, sheet_name=f'test{self.episode}', index=True)
        # self.recorder.excel_writer.save()

    def get_sample_data(self, index):
        i_data = self.data.iloc[index:index + self.batch_size]
        s = np.vstack(i_data.s.values).astype(np.float32)
        visual_s = np.vstack(i_data.visual_s.values).astype(np.float32)
        a = np.vstack(i_data.a.values).astype(np.float32)
        dc_r = np.vstack(i_data.discounted_reward.values).reshape(-1, 1).astype(np.float32)
        old_log_prob = np.vstack(i_data.log_prob.values).astype(np.float32)
        advantage = np.vstack(i_data.advantage.values).reshape(-1, 1).astype(np.float32)
        return s, visual_s, a, dc_r, old_log_prob, advantage

    # @show_graph(name='ppo_net')
    def learn(self, **kwargs):
        assert self.batch_size <= self.data.shape[0], "batch_size must less than the length of an episode"
        self.episode = kwargs['episode']
        self.calculate_statistics()
        for _ in range(self.epoch):
            for index in range(0, self.data.shape[0], self.batch_size):
                s, visual_s, a, dc_r, old_log_prob, advantage = map(tf.convert_to_tensor, self.get_sample_data(index))
                if self.share_net:
                    actor_loss, critic_loss, entropy, kl = self.train_share.get_concrete_function(
                        *self.TensorSpecs)(s, visual_s, a, dc_r, old_log_prob, advantage)
                else:
                    actor_loss, entropy, kl = self.train_actor.get_concrete_function(
                        *self.actor_TensorSpecs)(s, visual_s, a, old_log_prob, advantage)
                    # if kl > 1.5 * 0.01:
                    #     break
                    critic_loss = self.train_critic.get_concrete_function(
                        *self.critic_TensorSpecs)(s, visual_s, dc_r)
        self.global_step.assign_add(1)
        summaries = dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/critic_loss', critic_loss],
            ['Statistics/entropy', entropy]
        ])
        if self.share_net:
            summaries.update(dict([['LEARNING_RATE/lr', self.lr(self.episode)]]))
        else:
            summaries.update(dict([
                ['LEARNING_RATE/actor_lr', self.actor_lr(self.episode)],
                ['LEARNING_RATE/critic_lr', self.critic_lr(self.episode)]
            ]))
        self.write_training_summaries(self.episode, summaries)
        self.clear()

    @tf.function(experimental_relax_shapes=True)
    def train_share(self, s, visual_s, a, dc_r, old_log_prob, advantage):
        s, visual_s, a, dc_r, old_log_prob, advantage = self.cast(s, visual_s, a, dc_r, old_log_prob, advantage)
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu, value = self.net(s, visual_s)
                    new_log_prob = gaussian_likelihood_sum(mu, a, self.log_std)
                    entropy = gaussian_entropy(self.log_std)
                else:
                    logits, value = self.net(s, visual_s)
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
                loss = -(actor_loss - 1.0 * value_loss + self.beta * entropy)
            if self.is_continuous:
                loss_grads = tape.gradient(loss, self.net.tv)
                self.optimizer.apply_gradients(
                    zip(loss_grads, self.net.tv)
                )
            else:
                loss_grads = tape.gradient(loss, self.net.tv)
                self.optimizer.apply_gradients(
                    zip(loss_grads, self.net.tv)
                )
            return actor_loss, value_loss, entropy, kl

    @tf.function(experimental_relax_shapes=True)
    def train_actor(self, s, visual_s, a, old_log_prob, advantage):
        s, visual_s, a, old_log_prob, advantage = self.cast(s, visual_s, a, old_log_prob, advantage)
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu = self.actor_net(s, visual_s)
                    new_log_prob = gaussian_likelihood_sum(mu, a, self.log_std)
                    entropy = gaussian_entropy(self.log_std)
                else:
                    logits = self.actor_net(s, visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    new_log_prob = tf.reduce_sum(a * logp_all, axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                ratio = tf.exp(new_log_prob - old_log_prob)
                kl = tf.reduce_mean(old_log_prob - new_log_prob)
                surrogate = ratio * advantage
                min_adv = tf.where(advantage > 0, (1 + self.epsilon) * advantage, (1 - self.epsilon) * advantage)
                actor_loss = -(tf.reduce_mean(tf.minimum(surrogate, min_adv)) + self.beta * entropy)
            if self.is_continuous:
                actor_grads = tape.gradient(actor_loss, self.actor_net.tv)
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_net.tv)
                )
            else:
                actor_grads = tape.gradient(actor_loss, self.actor_net.tv)
                self.optimizer_actor.apply_gradients(
                    zip(actor_grads, self.actor_net.tv)
                )
            return actor_loss, entropy, kl

    @tf.function(experimental_relax_shapes=True)
    def train_critic(self, s, visual_s, dc_r):
        s, visual_s, dc_r = self.cast(s, visual_s, dc_r)
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                value = self.critic_net(s, visual_s)
                td_error = dc_r - value
                value_loss = tf.reduce_mean(tf.square(td_error))
            critic_grads = tape.gradient(value_loss, self.critic_net.tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_net.tv)
            )
            return value_loss
