import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import Nn
from utils.sth import sth
from .policy import Policy


class PG(Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 action_type,
                 lr=5.0e-4,
                 gamma=0.99,
                 max_episode=50000,
                 batch_size=100,
                 epoch=5,
                 base_dir=None,
                 logger2file=False,
                 out_graph=False):
        super().__init__(s_dim, visual_sources, visual_resolution, a_dim_or_list, action_type, gamma, max_episode, base_dir, 'ON')
        self.epoch = epoch
        self.batch_size = batch_size
        self.sigma_offset = np.full([self.a_counts, ], 0.01)
        self.lr = lr
        if self.action_type == 'continuous':
            self.net = Nn.actor_continuous(self.s_dim, self.visual_dim, self.a_counts, 'pg')
        else:
            self.net = Nn.actor_discrete(self.s_dim, self.visual_dim, self.a_counts, 'pg')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.generate_recorder(
            logger2file=logger2file,
            model=self
        )
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

    def choose_action(self, s, visual_s):
        if self.action_type == 'continuous':
            return self._get_action(s, visual_s).numpy()
        else:
            if np.random.uniform() < 0.2:
                a = np.random.randint(0, self.a_counts, len(s))
            else:
                a = self._get_action(s, visual_s).numpy()
            return sth.int2action_index(a, self.a_dim_or_list)

    def choose_inference_action(self, s, visual_s):
        a = self._get_action(s, visual_s).numpy()
        return a if self.action_type == 'continuous' else sth.int2action_index(a, self.a_dim_or_list)

    @tf.function
    def _get_action(self, vector_input, visual_input):
        with tf.device(self.device):
            if self.action_type == 'continuous':
                mu, sigma = self.net(vector_input, visual_input)
                norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma + self.sigma_offset)
                sample_op = tf.clip_by_value(norm_dist.sample(), -1, 1)
            else:
                action_probs = self.net(vector_input, visual_input)
                sample_op = tf.argmax(action_probs, axis=1)
        return sample_op

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        self.on_store(s, visual_s, a, r, s_, visual_s_, done)

    def calculate_statistics(self):
        self.data['total_reward'] = sth.discounted_sum(self.data.r.values, 1, 0, self.data.done.values)
        a = np.array(sth.discounted_sum(self.data.r.values, self.gamma, 0, self.data.done.values))
        a -= np.mean(a)
        a /= np.std(a)
        self.data['discounted_reward'] = list(a)

    def get_sample_data(self):
        i_data = self.data.sample(n=self.batch_size) if self.batch_size < self.data.shape[0] else self.data
        s = np.vstack([i_data.s.values[i] for i in range(i_data.shape[0])])
        visual_s = np.vstack([i_data.visual_s.values[i] for i in range(i_data.shape[0])])
        a = np.vstack([i_data.a.values[i] for i in range(i_data.shape[0])])
        dc_r = np.vstack([i_data.discounted_reward.values[i][:, np.newaxis] for i in range(i_data.shape[0])])
        return s, visual_s, a, dc_r

    def learn(self, episode):
        self.calculate_statistics()
        for _ in range(self.epoch):
            s, visual_s, a, dc_r = self.get_sample_data()
            self.global_step.assign_add(1)
            if not self.action_type == 'continuous':
                a = th.action_index2one_hot(a, self.a_dim_or_list)
            loss, entropy = self.train(s, visual_s, a, dc_r)
            tf.summary.experimental.set_step(self.global_step)
            if entropy is not None:
                tf.summary.scalar('LOSS/entropy', entropy)
            tf.summary.scalar('LOSS/loss', loss)
            tf.summary.scalar('LEARNING_RATE/lr', self.lr)
            tf.summary.scalar('REWARD/discounted_reward', self.data.discounted_reward.values[0].mean())
            tf.summary.scalar('REWARD/reward', self.data.total_reward.values[0].mean())
            self.recorder.writer.flush()
        self.clear()

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, dc_r):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                if self.action_type == 'continuous':
                    mu, sigma = self.net(s, visual_s)
                    norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma + self.sigma_offset)
                    sample_op = tf.clip_by_value(norm_dist.sample(), -1, 1)
                    log_act_prob = tf.reduce_mean(norm_dist.log_prob(a), axis=1)
                    entropy = tf.reduce_mean(norm_dist.entropy())
                else:
                    action_probs = self.net(s, visual_s)
                    sample_op = tf.argmax(action_probs, axis=1)
                    log_act_prob = tf.log(tf.reduce_sum(tf.multiply(action_probs, a), axis=1), keepdims=True)
                loss = tf.reduce_mean(log_act_prob * dc_r)
            loss_grads = tape.gradient(loss, self.net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(loss_grads, self.net.trainable_variables)
            )
            return loss, entropy if self.action_type == 'continuous' else None
