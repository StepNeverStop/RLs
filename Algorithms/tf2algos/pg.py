import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import Nn
from utils.sth import sth
from utils.tf2_utils import get_TensorSpecs, gaussian_clip_rsample, gaussian_likelihood_sum, gaussian_entropy
from Algorithms.tf2algos.base.on_policy import On_Policy


class PG(On_Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,

                 lr=5.0e-4,
                 epoch=5,
                 hidden_units={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32]
                 },
                 **kwargs):
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim_or_list=a_dim_or_list,
            is_continuous=is_continuous,
            **kwargs)
        self.epoch = epoch
        self.TensorSpecs = get_TensorSpecs([self.s_dim], self.visual_dim, [self.a_counts], [1])
        self.visual_net = Nn.VisualNet('visual_net', self.visual_dim)
        if self.is_continuous:
            self.net = Nn.actor_mu(self.s_dim, self.visual_dim, self.a_counts, 'pg_net', hidden_units['actor_continuous'], visual_net=self.actor_visual_net)
            self.log_std = tf.Variable(initial_value=-0.5 * np.ones(self.a_counts, dtype=np.float32), trainable=True)
            self.net.tv+=[self.log_std]
        else:
            self.net = Nn.actor_discrete(self.s_dim, self.visual_dim, self.a_counts, 'pg_net', hidden_units['actor_discrete'], visual_net=self.actor_visual_net)
        self.lr = tf.keras.optimizers.schedules.PolynomialDecay(lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr(self.episode))
    
    def show_logo(self):
        self.recorder.logger.info('''
　　　ｘｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　　　ｘｘ　　ｘｘ　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　　　　　　　
　　　　　ｘｘｘｘｘｘ　　　　　　　ｘ　　　ｘｘｘｘｘ　　　
　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　ｘｘｘ　　　　
　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　　ｘ　　　　　　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　　ｘｘｘｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　　　　　
        ''')

    def choose_action(self, s, visual_s, evaluation=False):
        a = self._get_action(s, visual_s, evaluation).numpy()
        return a if self.is_continuous else sth.int2action_index(a, self.a_dim_or_list)

    @tf.function
    def _get_action(self, s, visual_s, evaluation):
        s, visual_s = self.cast(s, visual_s)
        with tf.device(self.device):
            if self.is_continuous:
                mu = self.net(s, visual_s)
                sample_op, _ = gaussian_clip_rsample(mu, self.log_std)
            else:
                logits = self.net(s, visual_s)
                norm_dist = tfp.distributions.Categorical(logits)
                sample_op = norm_dist.sample()
        return sample_op

    def calculate_statistics(self):
        self.data['total_reward'] = sth.discounted_sum(self.data.r.values, 1, 0, self.data.done.values)
        a = np.asarray(sth.discounted_sum(self.data.r.values, self.gamma, 0, self.data.done.values))
        a -= np.mean(a)
        a /= np.std(a)
        self.data['discounted_reward'] = list(a)

    def get_sample_data(self, index):
        i_data = self.data.iloc[index:index + self.batch_size]
        s = np.vstack(i_data.s.values).astype(np.float32)
        visual_s = np.vstack(i_data.visual_s.values).astype(np.float32)
        a = np.vstack(i_data.a.values).astype(np.float32)
        dc_r = np.vstack(i_data.discounted_reward.values).reshape(-1, 1).astype(np.float32)
        return s, visual_s, a, dc_r

    def learn(self, **kwargs):
        assert self.batch_size <= self.data.shape[0], "batch_size must less than the length of an episode"
        self.episode = kwargs['episode']
        self.calculate_statistics()
        for _ in range(self.epoch):
            for index in range(0, self.data.shape[0], self.batch_size):
                s, visual_s, a, dc_r = map(tf.convert_to_tensor, self.get_sample_data(index))
                loss, entropy = self.train.get_concrete_function(
                    *self.TensorSpecs)(s, visual_s, a, dc_r)
        self.write_training_summaries(self.episode, dict([
            ['LOSS/loss', loss],
            ['Statistics/entropy', entropy],
            ['LEARNING_RATE/lr', self.lr(self.episode)]
        ]))
        self.clear()

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, dc_r):
        s, visual_s, a, dc_r = self.cast(s, visual_s, a, dc_r)
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu = self.net(s, visual_s)
                    log_act_prob = gaussian_likelihood_sum(mu, a, self.log_std)
                    entropy = gaussian_entropy(self.log_std)
                else:
                    logits = self.net(s, visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    log_act_prob = tf.reduce_sum(tf.multiply(logp_all, a), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                loss = tf.reduce_mean(log_act_prob * dc_r)
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
            self.global_step.assign_add(1)
            return loss, entropy
