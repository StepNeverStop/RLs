import numpy as np
import tensorflow as tf
from utils.sth import sth
from Algorithms.algorithm_base import Policy

initKernelAndBias = {
    'kernel_initializer': tf.random_normal_initializer(0., .1),
    'bias_initializer': tf.constant_initializer(0.1, dtype=tf.float32)
}


class PG(Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolutions,
                 a_dim_or_list,
                 action_type,
                 lr=5.0e-4,
                 gamma=0.99,
                 max_episode=50000,
                 batch_size=100,
                 epoch=5,
                 cp_dir=None,
                 log_dir=None,
                 excel_dir=None,
                 logger2file=False,
                 out_graph=False):
        super().__init__(s_dim, visual_sources, visual_resolutions, a_dim_or_list, action_type, max_episode, cp_dir, 'ON')
        self.epoch = epoch
        self.gamma = gamma
        self.batch_size = batch_size
        with self.graph.as_default():
            self.dc_r = tf.placeholder(tf.float32, [None, 1], name="discounted_reward")
            self.sigma_offset = tf.placeholder(tf.float32, [self.a_counts, ], 'sigma_offset')
            self.lr = tf.train.polynomial_decay(lr, self.episode, self.max_episode, 1e-10, power=1.0)
            if self.action_type == 'continuous':
                self.norm_dist = self._build_continuous_net('pg')
                self.sample_op = tf.clip_by_value(self.norm_dist.sample(), -1, 1)
                log_act_prob = tf.reduce_mean(self.norm_dist.log_prob(self.pl_a), axis=1)
                self.entropy = self.norm_dist.entropy()
                tf.summary.scalar('LOSS/entropy', tf.reduce_mean(self.entropy))
            else:
                self.action_multiplication_factor = sth.get_action_multiplication_factor(self.a_dim_or_list)
                self._build_discrete_net('pg')
                self.sample_op = tf.argmax(self.action_probs, axis=1)
                log_act_prob = tf.log(tf.reduce_sum(tf.multiply(self.action_probs, self.pl_a), axis=1))[:, np.newaxis]

            self.action = tf.identity(self.sample_op, name='action')
            net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pg')
            self.loss = tf.reduce_mean(log_act_prob * self.dc_r)
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.loss, var_list=net_vars + self.conv_vars, global_step=self.global_step)

            tf.summary.scalar('LOSS/loss', tf.reduce_mean(self.loss))
            tf.summary.scalar('LEARNING_RATE/lr', tf.reduce_mean(self.lr))
            self.summaries = tf.summary.merge_all()
            self.generate_recorder(
                cp_dir=cp_dir,
                log_dir=log_dir,
                excel_dir=excel_dir,
                logger2file=logger2file,
                graph=self.graph if out_graph else None
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
            self.init_or_restore(cp_dir)

    def _build_discrete_net(self, name):
        with tf.variable_scope(name):
            actor1 = tf.layers.dense(
                inputs=self.s,
                units=128,
                activation=self.activation_fn,
                name='actor1',
                **initKernelAndBias
            )
            actor2 = tf.layers.dense(
                inputs=actor1,
                units=64,
                activation=self.activation_fn,
                name='actor2',
                **initKernelAndBias
            )
            self.action_probs = tf.layers.dense(
                inputs=actor2,
                units=self.a_counts,
                activation=tf.nn.softmax,
                name='action_probs',
                **initKernelAndBias
            )

    def _build_continuous_net(self, name):
        with tf.variable_scope(name):
            actor1 = tf.layers.dense(
                inputs=self.s,
                units=128,
                activation=self.activation_fn,
                name='actor1',
                **initKernelAndBias
            )
            actor2 = tf.layers.dense(
                inputs=actor1,
                units=64,
                activation=self.activation_fn,
                name='actor2',
                **initKernelAndBias
            )
            self.mu = tf.layers.dense(
                inputs=actor2,
                units=self.a_counts,
                activation=tf.nn.tanh,
                name='mu',
                **initKernelAndBias
            )
            sigma1 = tf.layers.dense(
                inputs=actor1,
                units=64,
                activation=self.activation_fn,
                name='sigma1',
                **initKernelAndBias
            )
            self.sigma = tf.layers.dense(
                inputs=sigma1,
                units=self.a_counts,
                activation=tf.nn.sigmoid,
                name='sigma',
                **initKernelAndBias
            )
        norm_dist = tf.distributions.Normal(loc=self.mu, scale=self.sigma + self.sigma_offset)
        return norm_dist

    def choose_action(self, s):
        if self.action_type == 'continuous':
            pl_visual_s, pl_s = self.get_visual_and_vector_input(s)
            return self.sess.run(self.action, feed_dict={
                self.pl_visual_s: pl_visual_s,
                self.pl_s: pl_s,
                self.sigma_offset: np.full(self.a_counts, 0.01)
            })
        else:
            if np.random.uniform() < 0.2:
                a = np.random.randint(0, self.a_counts, len(s))
            else:
                pl_visual_s, pl_s = self.get_visual_and_vector_input(s)
                a = self.sess.run(self.action, feed_dict={
                    self.pl_visual_s: pl_visual_s,
                    self.pl_s: pl_s
                })
            return sth.int2action_index(a, self.action_multiplication_factor)

    def choose_inference_action(self, s):
        pl_visual_s, pl_s = self.get_visual_and_vector_input(s)
        a = self.sess.run(self.action, feed_dict={
            self.pl_visual_s: pl_visual_s,
            self.pl_s: pl_s,
            self.sigma_offset: np.full(self.a_counts, 0.01)
        })
        return a if self.action_type == 'continuous' else sth.int2action_index(a, self.action_multiplication_factor)

    def store_data(self, s, a, r, s_, done):
        self.on_store(s, a, r, s_, done)

    def calculate_statistics(self):
        self.data['total_reward'] = sth.discounted_sum(self.data.r.values, 1, 0, self.data.done.values)
        a = np.array(sth.discounted_sum(self.data.r.values, self.gamma, 0, self.data.done.values))
        a -= np.mean(a)
        a /= np.std(a)
        self.data['discounted_reward'] = list(a)

    def get_sample_data(self):
        i_data = self.data.sample(n=self.batch_size) if self.batch_size < self.data.shape[0] else self.data
        s = np.vstack([i_data.s.values[i] for i in range(i_data.shape[0])])
        a = np.vstack([i_data.a.values[i] for i in range(i_data.shape[0])])
        dc_r = np.vstack([i_data.discounted_reward.values[i][:, np.newaxis] for i in range(i_data.shape[0])])
        return s, a, dc_r

    def learn(self, episode):
        self.calculate_statistics()
        for _ in range(self.epoch):
            s, a, dc_r = self.get_sample_data()
            pl_visual_s, pl_s = self.get_visual_and_vector_input(s)
            summaries, _ = self.sess.run([self.summaries, self.train_op], feed_dict={
                self.pl_visual_s: pl_visual_s,
                self.pl_s: pl_s,
                self.pl_a: a if self.action_type == 'continuous' else sth.get_batch_one_hot(a, self.action_multiplication_factor, self.a_counts),
                self.dc_r: dc_r,
                self.episode: episode,
                self.sigma_offset: np.full(self.a_counts, 0.01)
            })
            self.recorder.writer.add_summary(summaries, self.sess.run(self.global_step))
        self.recorder.writer_summary(
            x=episode,
            ys=[{
                'tag': 'REWARD/discounted_reward',
                'value': self.data.discounted_reward.values[0].mean()
            },
                {
                'tag': 'REWARD/reward',
                'value': self.data.total_reward.values[0].mean()
            }])
        self.clear()
