import numpy as np
import tensorflow as tf
import Nn
from utils.sth import sth
from Algorithms.algorithm_base import Policy

class A2C(Policy):
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
                 cp_dir=None,
                 log_dir=None,
                 excel_dir=None,
                 logger2file=False,
                 out_graph=False):
        super().__init__(s_dim, visual_sources, visual_resolutions, a_dim_or_list, action_type, gamma, max_episode, cp_dir, 'ON')
        self.batch_size = batch_size
        with self.graph.as_default():
            self.dc_r = tf.placeholder(tf.float32, [None, 1], name="discounted_reward")
            self.lr = tf.train.polynomial_decay(lr, self.episode, self.max_episode, 1e-10, power=1.0)
            self.sigma_offset = tf.placeholder(tf.float32, [self.a_counts, ], 'sigma_offset')
            if self.action_type == 'continuous':
                self.mu, self.sigma = Nn.actor_continuous('actor', self.s, self.a_counts)
                self.norm_dist = tf.distributions.Normal(loc=self.mu, scale=self.sigma + self.sigma_offset)
                self.sample_op = tf.clip_by_value(self.norm_dist.sample(), -1, 1)
                log_act_prob = self.norm_dist.log_prob(self.pl_a)
                self.v = Nn.critic_v('critic', self.s)
                self.entropy = self.norm_dist.entropy()
                tf.summary.scalar('LOSS/entropy', tf.reduce_mean(self.entropy))
            else:
                self.action_probs = Nn.actor_discrete('actor', self.s)
                self.v = Nn.critic_v('critic', self.s)
                self.sample_op = tf.argmax(self.action_probs, axis=1)
                log_act_prob = tf.log(tf.reduce_sum(tf.multiply(self.action_probs, self.pl_a), axis=1))[:, np.newaxis]
            self.action = tf.identity(self.sample_op, name='action')

            self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
            self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

            self.advantage = tf.stop_gradient(self.dc_r - self.v)
            self.actor_loss = tf.reduce_mean(log_act_prob * self.advantage)
            self.critic_loss = tf.reduce_mean(tf.squared_difference(self.v, self.dc_r))
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_critic = optimizer.minimize(self.critic_loss, var_list=self.critic_vars + self.conv_vars)
            with tf.control_dependencies([self.train_critic]):
                self.train_actor = optimizer.minimize(-self.actor_loss, var_list=self.actor_vars + self.conv_vars, global_step=self.global_step)
            self.train_sequence = [self.train_critic, self.train_actor]

            tf.summary.scalar('LOSS/actor_loss', tf.reduce_mean(-self.actor_loss))
            tf.summary.scalar('LOSS/critic_loss', tf.reduce_mean(self.critic_loss))
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
　　　　　　　ｘｘ　　　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘ　　　　
　　　　　　ｘｘｘ　　　　　　　　　　　ｘｘ　ｘｘｘ　　　　　　　　ｘｘｘ　　ｘｘ　　　　
　　　　　　ｘｘｘ　　　　　　　　　　　ｘｘ　ｘｘｘ　　　　　　　　ｘｘ　　　　ｘｘ　　　
　　　　　　ｘ　ｘｘ　　　　　　　　　　　　　ｘｘ　　　　　　　　　ｘｘ　　　　　　　　　
　　　　　ｘｘ　ｘｘ　　　　　　　　　　　　ｘｘｘ　　　　　　　　ｘｘｘ　　　　　　　　　
　　　　　ｘｘｘｘｘｘ　　　　　　　　　　　ｘｘ　　　　　　　　　ｘｘｘ　　　　　　　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　　　ｘｘ　　　　　　　　　　　ｘｘ　　　　ｘｘ　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　　ｘｘ　　ｘ　　　　　　　　　ｘｘｘ　　ｘｘｘ　　　
　　　ｘｘｘ　　ｘｘｘｘｘ　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘ　　　　
            ''')
            self.init_or_restore(cp_dir)

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
            return sth.int2action_index(a, self.a_dim_or_list)

    def choose_inference_action(self, s):
        pl_visual_s, pl_s = self.get_visual_and_vector_input(s)
        if self.action_type == 'continuous':
            return self.sess.run(self.mu, feed_dict={
                self.pl_visual_s: pl_visual_s,
                self.pl_s: pl_s,
                self.sigma_offset: np.full(self.a_counts, 0.01)
            })
        else:
            a = self.sess.run(self.action, feed_dict={
                self.pl_visual_s: pl_visual_s,
                self.pl_s: pl_s,
                self.sigma_offset: np.full(self.a_counts, 0.01)
            })
            return sth.int2action_index(a, self.a_dim_or_list)

    def store_data(self, s, a, r, s_, done):
        self.on_store(s, a, r, s_, done)

    def calculate_statistics(self):
        pl_visual_s, pl_s = self.get_visual_and_vector_input(self.data.s_.values[-1])
        init_value = np.squeeze(self.sess.run(self.v, feed_dict={
            self.pl_visual_s: pl_visual_s,
            self.pl_s: pl_s,
        }))
        self.data['discounted_reward'] = sth.discounted_sum(self.data.r.values, self.gamma, init_value, self.data.done.values)

    def get_sample_data(self):
        i_data = self.data.sample(n=self.batch_size) if self.batch_size < self.data.shape[0] else self.data
        s = np.vstack([i_data.s.values[i] for i in range(i_data.shape[0])])
        a = np.vstack([i_data.a.values[i] for i in range(i_data.shape[0])])
        dc_r = np.vstack([i_data.discounted_reward.values[i][:, np.newaxis] for i in range(i_data.shape[0])])
        return s, a, dc_r

    def learn(self, episode):
        self.calculate_statistics()
        s, a, dc_r = self.get_sample_data()
        pl_visual_s, pl_s = self.get_visual_and_vector_input(s)
        summaries, _ = self.sess.run([self.summaries, self.train_sequence], feed_dict={
            self.pl_visual_s: pl_visual_s,
            self.pl_s: pl_s,
            self.pl_a: a if self.action_type == 'continuous' else sth.action_index2one_hot(a, self.a_dim_or_list),
            self.dc_r: dc_r,
            self.episode: episode,
            self.sigma_offset: np.full(self.a_counts, 0.01)
        })
        self.recorder.writer.add_summary(summaries, self.sess.run(self.global_step))
        self.clear()
