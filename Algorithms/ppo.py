import numpy as np
import tensorflow as tf
from utils.sth import sth
from Algorithms.algorithm_base import Policy


initKernelAndBias = {
    'kernel_initializer': tf.random_normal_initializer(0., .1),
    'bias_initializer': tf.constant_initializer(0.1, dtype=tf.float32)
}


class PPO(Policy):
    def __init__(self,
                 s_dim,
                 a_counts,
                 action_type,
                 epsilon=0.2,
                 gamma=0.99,
                 beta=1.0e-3,
                 lr=5.0e-4,
                 lambda_=0.95,
                 max_episode=50000,
                 batch_size=100,
                 epoch=5,
                 cp_dir=None,
                 log_dir=None,
                 excel_dir=None,
                 logger2file=False,
                 out_graph=False):
        super().__init__(s_dim, a_counts, action_type, max_episode, cp_dir, 'ON')
        self.epsilon = epsilon
        self.gamma = gamma
        self.beta = beta
        self.epoch = epoch
        self.lambda_ = lambda_
        self.batch_size = batch_size
        with self.graph.as_default():
            self.dc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_reward')
            self.advantage = tf.placeholder(tf.float32, [None, 1], "advantage")
            self.sigma_offset = tf.placeholder(tf.float32, [self.a_counts, ], 'sigma_offset')
            self.norm_dist = self._build_net('ppo')
            self.old_prob = tf.placeholder(tf.float32, [None, self.a_counts], 'old_prob')

            self.new_prob = self.norm_dist.prob(self.pl_a)

            self.sample_op = tf.clip_by_value(self.norm_dist.sample(), -1, 1)
            self.action = tf.identity(self.sample_op, name='action')
            self.entropy = self.norm_dist.entropy()
            # ratio = tf.exp(self.new_prob - self.old_prob)
            ratio = self.new_prob / self.old_prob
            surrogate = ratio * self.advantage
            self.actor_loss = tf.reduce_mean(
                tf.minimum(
                    surrogate,
                    tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * self.advantage
                ))
            self.value_loss = tf.reduce_mean(tf.squared_difference(self.dc_r, self.value))
            self.loss = -(self.actor_loss - 1.0 * self.value_loss + self.beta * tf.reduce_mean(self.entropy))
            self.lr = tf.train.polynomial_decay(lr, self.episode, self.max_episode, 1e-10, power=1.0)
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)
            tf.summary.scalar('LOSS/actor_loss', tf.reduce_mean(self.actor_loss))
            tf.summary.scalar('LOSS/critic_loss', tf.reduce_mean(self.value_loss))
            tf.summary.scalar('LOSS/entropy', tf.reduce_mean(self.entropy))
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
            self.init_or_restore(cp_dir)

    def _build_net(self, name):
        with tf.variable_scope(name):
            share1 = tf.layers.dense(
                inputs=self.pl_s,
                units=512,
                activation=self.activation_fn,
                name='share1',
                **initKernelAndBias
            )
            share2 = tf.layers.dense(
                inputs=share1,
                units=256,
                activation=self.activation_fn,
                name='share2',
                **initKernelAndBias
            )
            actor1 = tf.layers.dense(
                inputs=share2,
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
            critic1 = tf.layers.dense(
                inputs=share2,
                units=128,
                activation=self.activation_fn,
                name='critic1',
                **initKernelAndBias
            )
            critic2 = tf.layers.dense(
                inputs=critic1,
                units=64,
                activation=self.activation_fn,
                name='critic2',
                **initKernelAndBias
            )
            self.value = tf.layers.dense(
                inputs=critic2,
                units=1,
                activation=None,
                name='value',
                **initKernelAndBias
            )
        norm_dist = tf.distributions.Normal(loc=self.mu, scale=self.sigma + self.sigma_offset)
        return norm_dist

    def choose_action(self, s):
        return self.sess.run(self.action, feed_dict={
            self.pl_s: s,
            self.sigma_offset: np.full(self.a_counts, 0.01)
        })

    def choose_inference_action(self, s):
        return self.sess.run(self.action, feed_dict={
            self.pl_s: s,
            self.sigma_offset: np.full(self.a_counts, 0.01)
        })

    def store_data(self, s, a, r, s_, done):
        assert isinstance(s, np.ndarray)
        assert isinstance(a, np.ndarray)
        assert isinstance(r, np.ndarray)
        assert isinstance(s_, np.ndarray)
        assert isinstance(done, np.ndarray)

        self.data = self.data.append({
            's': s,
            'a': a,
            'r': r,
            's_': s_,
            'done': done,
            'value': np.squeeze(self.sess.run(self.value, feed_dict={
                self.pl_s: s,
                self.sigma_offset: np.full(self.a_counts, 0.01)
            })),
            'next_value': np.squeeze(self.sess.run(self.value, feed_dict={
                self.pl_s: s_,
                self.sigma_offset: np.full(self.a_counts, 0.01)
            })),
            'prob': self.sess.run(self.new_prob, feed_dict={
                self.pl_s: s,
                self.pl_a: a,
                self.sigma_offset: np.full(self.a_counts, 0.01)
            }) + 1e-10
        }, ignore_index=True)

    def calculate_statistics(self):
        self.data['total_reward'] = sth.discounted_sum(self.data.r.values, 1, 0, self.data.done.values)
        self.data['discounted_reward'] = sth.discounted_sum(self.data.r.values, self.gamma, self.data.next_value.values[-1], self.data.done.values)
        self.data['td_error'] = sth.discounted_sum_minus(
            self.data.r.values,
            self.gamma,
            self.data.next_value.values[-1],
            self.data.done.values,
            self.data.value.values
        )
        self.data['advantage'] = sth.discounted_sum(
            self.data.td_error.values,
            self.lambda_ * self.gamma,
            0,
            self.data.done.values
        )
        # self.data.to_excel(self.excel_writer, sheet_name='test', index=True)
        # self.excel_writer.save()

    def get_sample_data(self):
        i_data = self.data.sample(n=self.batch_size) if self.batch_size < self.data.shape[0] else self.data
        s = np.vstack([i_data.s.values[i] for i in range(i_data.shape[0])])
        a = np.vstack([i_data.a.values[i] for i in range(i_data.shape[0])])
        dc_r = np.vstack([i_data.discounted_reward.values[i][:, np.newaxis] for i in range(i_data.shape[0])])
        old_prob = np.vstack([i_data.prob.values[i] for i in range(i_data.shape[0])])
        advantage = np.vstack([i_data.advantage.values[i][:, np.newaxis] for i in range(i_data.shape[0])])
        return s, a, dc_r, old_prob, advantage

    def learn(self, episode):
        self.calculate_statistics()
        for _ in range(self.epoch):
            s, a, dc_r, old_prob, advantage = self.get_sample_data()
            summaries, _ = self.sess.run([self.summaries, self.train_op], feed_dict={
                self.pl_s: s,
                self.pl_a: a,
                self.dc_r: dc_r,
                self.old_prob: old_prob,
                self.advantage: advantage,
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
