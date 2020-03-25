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


        if self.is_continuous:
            self.actor_net = Nn.actor_mu(self.rnn_net.hdim, self.a_counts, hidden_units['actor_continuous'])
            self.log_std = tf.Variable(initial_value=-0.5 * np.ones(self.a_counts, dtype=np.float32), trainable=True)
            self.actor_tv = self.actor_net.trainable_variables + [self.log_std]
            self.Hx_TensorSpecs = [tf.TensorSpec(shape=flat_concat(self.actor_tv).shape, dtype=tf.float32)] \
                + get_TensorSpecs([self.s_dim], self.visual_dim, [self.a_counts], [self.a_counts])
        else:
            self.actor_net = Nn.actor_discrete(self.rnn_net.hdim, self.a_counts, hidden_units['actor_discrete'])
            self.actor_tv = self.actor_net.trainable_variables
            self.Hx_TensorSpecs = [tf.TensorSpec(shape=flat_concat(self.actor_tv).shape, dtype=tf.float32)] \
                + get_TensorSpecs([self.s_dim], self.visual_dim, [self.a_counts])
        self.critic_net = Nn.critic_v(self.rnn_net.hdim, hidden_units['critic'])
        self.critic_tv = self.critic_net.trainable_variables + self.other_tv
        self.critic_lr = tf.keras.optimizers.schedules.PolynomialDecay(critic_lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer_critic = tf.keras.optimizers.Adam(learning_rate=self.critic_lr(self.episode))

        self.model_recorder(dict(
            actor=self.actor_net,
            critic=self.critic_net,
            optimizer_critic=self.optimizer_critic
            ))

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
        feat, self.cell_state = self.get_feature(s, visual_s, self.cell_state, record_cs=True, train=False)
        a, _v, _lp, _morlpa = self._get_action(feat)
        a = a.numpy()
        self._value = np.squeeze(_v.numpy())
        self._log_prob = np.squeeze(_lp.numpy()) + 1e-10
        if self.is_continuous:
            self._mu = _morlpa.numpy()
        else:
            self._logp_all = _morlpa.numpy()
        return a if self.is_continuous else sth.int2action_index(a, self.a_dim_or_list)

    @tf.function
    def _get_action(self, s, visual_s):
        with tf.device(self.device):
            value = self.critic_net(feat)
            if self.is_continuous:
                mu = self.actor_net(feat)
                sample_op, _ = gaussian_clip_rsample(mu, self.log_std)
                log_prob = gaussian_likelihood_sum(mu, sample_op, self.log_std)
                return sample_op, value, log_prob, mu
            else:
                logits = self.actor_net(feat)
                logp_all = tf.nn.log_softmax(logits)
                norm_dist = tfp.distributions.Categorical(logits)
                sample_op = norm_dist.sample()
                log_prob = norm_dist.log_prob(sample_op)
                return sample_op, value, log_prob, logp_all

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
            's_': s_,
            'visual_s_': visual_s_,
            'done': done,
            'value': self._value,
            'log_prob': self._log_prob
        }
        if self.is_continuous:
            _data.update({'old_mu': self._mu})
            _data.update({'old_log_std': self.log_std.numpy()})
        else:
            _data.update({'old_logp_all': self._logp_all})
        self.data = self.data.append(_data, ignore_index=True)

    @tf.function
    def _get_value(self, feat):
        with tf.device(self.device):
            value = self.critic_net(feat)
            return value

    def calculate_statistics(self):
        feat, self.cell_state = self.get_feature(self.data.s_.values[-1], self.data.visual_s_.values[-1], self.cell_state, record_cs=True, train=False)
        init_value = np.squeeze(self._get_value(feat).numpy())
        # self.data['total_reward'] = sth.discounted_sum(self.data.r.values, 1, init_value, self.data.done.values)
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
        old_log_prob = np.vstack(i_data.log_prob.values).reshape(-1, 1).astype(np.float32)
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

        self.intermediate_variable_reset()
        if self.use_curiosity:
            ir, iloss, isummaries = self.curiosity_model(
                np.vstack(self.data.s.values).astype(np.float32),
                np.vstack(self.data.visual_s.values).astype(np.float32),
                np.vstack(self.data.a.values).astype(np.float32),
                np.vstack(self.data.s_.values).astype(np.float32),
                np.vstack(self.data.visual_s_.values).astype(np.float32)
                )
            r = np.vstack(self.data.r.values).reshape(-1) + np.squeeze(ir.numpy())
            self.data.r = list(r.reshape([self.data.shape[0], -1]))
            self.curiosity_loss_constant += iloss
            self.summaries.update(isummaries)

        self.calculate_statistics()
        for index in range(0, self.data.shape[0], self.batch_size):
            if self.is_continuous:
                s, visual_s, a, dc_r, old_log_prob, advantage, old_mu, old_log_std = map(self.data_convert, self.get_sample_data(index))
                Hx_args = (s, visual_s, old_mu, old_log_std)
            else:
                s, visual_s, a, dc_r, old_log_prob, advantage, old_logp_all = map(self.data_convert, self.get_sample_data(index))
                Hx_args = (s, visual_s, old_logp_all)
            actor_loss, entropy, gradients = self.train_actor.get_concrete_function(
                *self.actor_TensorSpecs)(s, visual_s, a, old_log_prob, advantage)

            x = self.cg(self.Hx.get_concrete_function(*self.Hx_TensorSpecs), gradients.numpy(), Hx_args)
            x = tf.convert_to_tensor(x)
            alpha = np.sqrt(2 * self.delta / (np.dot(x, self.Hx.get_concrete_function(*self.Hx_TensorSpecs)(x, *Hx_args)) + 1e-8))
            for i in range(self.backtrack_iters):
                assign_params_from_flat(alpha * x * (self.backtrack_coeff ** i), self.actor_tv)

            for _ in range(self.train_v_iters):
                critic_loss = self.train_critic.get_concrete_function(
                    *self.critic_TensorSpecs)(s, visual_s, dc_r)
        summaries = dict([
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/critic_loss', critic_loss],
            ['Statistics/entropy', entropy]
        ])
        summaries.update(dict([
            ['LEARNING_RATE/critic_lr', self.critic_lr(self.episode)]
        ]))
        self.summaries.update(summaries)
        self.write_training_summaries(self.episode, self.summaries)
        self.clear()

    @tf.function(experimental_relax_shapes=True)
    def train_actor(self, s, visual_s, a, old_log_prob, advantage):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat = self.get_feature(s, visual_s)
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
                actor_loss = -tf.reduce_mean(ratio * advantage)
            actor_grads = tape.gradient(actor_loss, self.actor_tv)
            gradients = flat_concat(actor_grads)
            self.global_step.assign_add(1)
            return actor_loss, entropy, gradients

    @tf.function(experimental_relax_shapes=True)
    def Hx(self, x, *args):
        if self.is_continuous:
            s, visual_s, old_mu, old_log_std = args
        else:
            s, visual_s, old_logp_all = args
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                feat = self.get_feature(s, visual_s)
                if self.is_continuous:
                    mu = self.actor_net(feat)
                    var0, var1 = tf.exp(2 * self.log_std), tf.exp(2 * old_log_std)
                    pre_sum = 0.5 * (((old_mu - mu)**2 + var0) / (var1 + 1e-8) - 1) + old_log_std - self.log_std
                    all_kls = tf.reduce_sum(pre_sum, axis=1)
                    kl = tf.reduce_mean(all_kls)
                else:
                    logits = self.actor_net(feat)
                    logp_all = tf.nn.log_softmax(logits)
                    all_kls = tf.reduce_sum(tf.exp(old_logp_all) * (old_logp_all - logp_all), axis=1)
                    kl = tf.reduce_mean(all_kls)

                g = flat_concat(tape.gradient(kl, self.actor_tv))
                _g = tf.reduce_sum(g * x)
                hvp = flat_concat(tape.gradient(_g, self.actor_tv))
                if self.damping_coeff > 0:
                    hvp += self.damping_coeff * x
            return hvp

    @tf.function(experimental_relax_shapes=True)
    def train_critic(self, s, visual_s, dc_r):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat = self.get_feature(s, visual_s)
                value = self.critic_net(feat)
                td_error = dc_r - value
                value_loss = tf.reduce_mean(tf.square(td_error)) + self.curiosity_loss_constant
            critic_grads = tape.gradient(value_loss, self.critic_tv)
            self.optimizer_critic.apply_gradients(
                zip(critic_grads, self.critic_tv)
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
