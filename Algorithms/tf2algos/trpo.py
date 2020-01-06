import Nn
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from utils.sth import sth
from utils.np_utils import normalization, standardization
from utils.tf2_utils import get_TensorSpecs, gaussian_clip_rsample, gaussian_likelihood_sum, gaussian_entropy
from Algorithms.tf2algos.base.on_policy import On_Policy
'''
Stole this from OpenAI SpinningUp. https://github.com/openai/spinningup/blob/master/spinup/algos/trpo/trpo.py
'''


def flat_concat(xs):
    return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)


def assign_params_from_flat(x, params):
    def flat_size(p): return int(np.prod(p.shape.as_list()))  # the 'int' is important for scalars
    splits = tf.split(x, [flat_size(p) for p in params])
    new_params = [tf.reshape(p_new, p.shape) for p, p_new in zip(params, splits)]
    return tf.group([p.assign(p_new) for p, p_new in zip(params, new_params)])


class TRPO(On_Policy):
    '''
    Trust Region Policy Optimization, https://arxiv.org/abs/1502.05477
    '''

    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,

                 beta=1.0e-3,
                 lr=5.0e-4,
                 delta=0.01,
                 lambda_=0.95,
                 cg_iters=10,
                 train_v_iters=10,
                 damping_coeff=0.1,
                 backtrack_iters=10,
                 backtrack_coeff=0.8,
                 share_visual_net=True,
                 epsilon=0.2,
                 critic_lr=1e-3,
                 hidden_units={
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
        self.delta = delta
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.cg_iters = cg_iters
        self.damping_coeff = damping_coeff
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.train_v_iters = train_v_iters

        self.actor_TensorSpecs = get_TensorSpecs([self.s_dim], self.visual_dim, [self.a_counts], [1], [1])
        self.critic_TensorSpecs = get_TensorSpecs([self.s_dim], self.visual_dim, [1])

        self.share_visual_net = share_visual_net
        if self.share_visual_net:
            self.actor_visual_net = self.critic_visual_net = Nn.VisualNet('visual_net', self.visual_dim)
        else:
            self.actor_visual_net = Nn.VisualNet('actor_visual_net', self.visual_dim)
            self.critic_visual_net = Nn.VisualNet('critic_visual_net', self.visual_dim)

        if self.is_continuous:
            self.actor_net = Nn.actor_mu(self.s_dim, self.a_counts, 'actor_net', hidden_units['actor_continuous'], visual_net=self.actor_visual_net)
            self.log_std = tf.Variable(initial_value=-0.5 * np.ones(self.a_counts, dtype=np.float32), trainable=True)
            self.actor_net.tv += [self.log_std]
            self.actor_params = self.actor_net.tv
            self.Hx_TensorSpecs = [tf.TensorSpec(shape=flat_concat(self.actor_params).shape, dtype=tf.float32)] \
                + get_TensorSpecs([self.s_dim], self.visual_dim, [self.a_counts], [self.a_counts])
        else:
            self.actor_net = Nn.actor_discrete(self.s_dim, self.a_counts, 'actor_net', hidden_units['actor_discrete'], visual_net=self.actor_visual_net)
            self.actor_params = self.actor_net.tv
            self.Hx_TensorSpecs = [tf.TensorSpec(shape=flat_concat(self.actor_params).shape, dtype=tf.float32)] \
                + get_TensorSpecs([self.s_dim], self.visual_dim, [self.a_counts])
        self.critic_net = Nn.critic_v(self.s_dim, 'critic_net', hidden_units['critic'], visual_net=self.critic_visual_net)
        self.critic_lr = tf.keras.optimizers.schedules.PolynomialDecay(critic_lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.critic_lr(self.episode))

    def show_logo(self):
        self.recorder.logger.info('''
　　　ｘｘｘｘｘｘｘｘｘ　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘ　　　　　
　　　ｘｘ　　ｘ　　ｘｘ　　　　　　　　ｘｘ　ｘｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　　　　　ｘｘｘ　ｘｘｘ　　　　
　　　ｘｘ　　ｘ　　ｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　ｘｘ　　　ｘｘ　　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　ｘｘ　　　ｘｘｘ　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　　ｘｘｘ　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘｘ　ｘｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　　ｘｘ　　　ｘｘｘ　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　　ｘｘ　　　ｘｘ　　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　　　　　　　　　　　ｘｘ　　ｘｘｘ　　　　
　　　　　ｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘ　ｘｘｘ　　　　　　ｘｘｘｘｘ　　　　　　　　　　　　ｘｘｘｘｘ　
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
                sample_op, _ = gaussian_clip_rsample(mu, self.log_std)
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
        _data = {
            's': s,
            'visual_s': visual_s,
            'a': a,
            'r': r,
            'done': done,
            'value': np.squeeze(self._get_value(s, visual_s).numpy()),
            'log_prob': self._get_log_prob(s, visual_s, a).numpy() + 1e-10
        }
        if self.is_continuous:
            _data.update({'old_mu': self.actor_net(s, visual_s).numpy()})
            _data.update({'old_log_std': self.log_std.numpy()})
        else:
            _data.update({'old_logp_all': tf.nn.log_softmax(self.actor_net(s, visual_s)).numpy()})
        self.data = self.data.append(_data, ignore_index=True)
        self.s_ = s_
        self.visual_s_ = visual_s_

    @tf.function
    def _get_value(self, s, visual_s):
        s, visual_s = self.cast(s, visual_s)
        with tf.device(self.device):
            value = self.critic_net(s, visual_s)
            return value

    @tf.function
    def _get_log_prob(self, s, visual_s, a):
        s, visual_s, a = self.cast(s, visual_s, a)
        with tf.device(self.device):
            if self.is_continuous:
                mu = self.actor_net(s, visual_s)
                new_log_prob = gaussian_likelihood_sum(mu, a, self.log_std)
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

    def get_sample_data(self, index):
        i_data = self.data.iloc[index:index + self.batch_size]
        s = np.vstack(i_data.s.values).astype(np.float32)
        visual_s = np.vstack(i_data.visual_s.values).astype(np.float32)
        a = np.vstack(i_data.a.values).astype(np.float32)
        dc_r = np.vstack(i_data.discounted_reward.values).reshape(-1, 1).astype(np.float32)
        old_log_prob = np.vstack(i_data.log_prob.values).astype(np.float32)
        advantage = np.vstack(i_data.advantage.values).reshape(-1, 1).astype(np.float32)
        if self.is_continuous:
            return (
                s, visual_s, a, dc_r, old_log_prob, advantage,
                np.vstack(i_data.old_mu.values).astype(np.float32),
                np.vstack(i_data.old_log_std.values).astype(np.float32)
            )
        else:
            return (
                s, visual_s, a, dc_r, old_log_prob, advantage,
                np.vstack(i_data.old_logp_all.values).astype(np.float32)
            )

    def learn(self, **kwargs):
        assert self.batch_size <= self.data.shape[0], "batch_size must less than the length of an episode"
        self.episode = kwargs['episode']
        self.calculate_statistics()
        for index in range(0, self.data.shape[0], self.batch_size):
            if self.is_continuous:
                s, visual_s, a, dc_r, old_log_prob, advantage, old_mu, old_log_std = map(tf.convert_to_tensor, self.get_sample_data(index))
                Hx_args = (s, visual_s, old_mu, old_log_std)
            else:
                s, visual_s, a, dc_r, old_log_prob, advantage, old_logp_all = map(tf.convert_to_tensor, self.get_sample_data(index))
                Hx_args = (s, visual_s, old_logp_all)
            actor_loss, entropy, gradients = self.train_actor.get_concrete_function(
                *self.actor_TensorSpecs)(s, visual_s, a, old_log_prob, advantage)

            x = self.cg(self.Hx.get_concrete_function(*self.Hx_TensorSpecs), gradients.numpy(), Hx_args)
            x = tf.convert_to_tensor(x)
            alpha = np.sqrt(2 * self.delta / (np.dot(x, self.Hx.get_concrete_function(*self.Hx_TensorSpecs)(x, *Hx_args)) + 1e-8))
            for i in range(self.backtrack_iters):
                assign_params_from_flat(alpha * x * (self.backtrack_coeff ** i), self.actor_params)

            for _ in range(self.train_v_iters):
                critic_loss = self.train_critic.get_concrete_function(
                    *self.critic_TensorSpecs)(s, visual_s, dc_r)
        self.global_step.assign_add(1)
        summaries = dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/critic_loss', critic_loss],
            ['Statistics/entropy', entropy]
        ])
        summaries.update(dict([
            ['LEARNING_RATE/critic_lr', self.critic_lr(self.episode)]
        ]))
        self.write_training_summaries(self.episode, summaries)
        self.clear()

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
                actor_loss = -tf.reduce_mean(ratio * advantage)
            actor_grads = tape.gradient(actor_loss, self.actor_params)
            gradients = flat_concat(actor_grads)
            return actor_loss, entropy, gradients

    @tf.function(experimental_relax_shapes=True)
    def Hx(self, x, *args):
        if self.is_continuous:
            s, visual_s, old_mu, old_log_std = self.cast(*args)
        else:
            s, visual_s, old_logp_all = self.cast(*args)
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                if self.is_continuous:
                    mu = self.actor_net(s, visual_s)
                    var0, var1 = tf.exp(2 * self.log_std), tf.exp(2 * old_log_std)
                    pre_sum = 0.5 * (((old_mu - mu)**2 + var0) / (var1 + 1e-8) - 1) + old_log_std - self.log_std
                    all_kls = tf.reduce_sum(pre_sum, axis=1)
                    kl = tf.reduce_mean(all_kls)
                else:
                    logits = self.actor_net(s, visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    all_kls = tf.reduce_sum(tf.exp(old_logp_all) * (old_logp_all - logp_all), axis=1)
                    kl = tf.reduce_mean(all_kls)

                g = flat_concat(tape.gradient(kl, self.actor_params))
                _g = tf.reduce_sum(g * x)
                hvp = flat_concat(tape.gradient(_g, self.actor_params))
                if self.damping_coeff > 0:
                    hvp += self.damping_coeff * x
            return hvp

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

    def cg(self, Ax, b, args):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """
        x = np.zeros_like(b)
        r = b.copy()  # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = r.copy()
        r_dot_old = np.dot(r, r)
        for _ in range(self.cg_iters):
            z = Ax(tf.convert_to_tensor(p), *args)
            alpha = r_dot_old / (np.dot(p, z) + 1e-8)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x
