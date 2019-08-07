import numpy as np
import tensorflow as tf
import Nn
from utils.sth import sth
from Algorithms.algorithm_base import Policy


class AC(Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolutions,
                 a_dim_or_list,
                 action_type,
                 lr=5.0e-4,
                 gamma=0.99,
                 max_episode=50000,
                 cp_dir=None,
                 log_dir=None,
                 excel_dir=None,
                 logger2file=False,
                 out_graph=False):
        super().__init__(s_dim, visual_sources, visual_resolutions, a_dim_or_list, action_type, gamma, max_episode, cp_dir, 'OFF')
        with self.graph.as_default():
            self.lr = tf.train.polynomial_decay(lr, self.episode, self.max_episode, 1e-10, power=1.0)
            self.sigma_offset = tf.placeholder(tf.float32, [self.a_counts, ], 'sigma_offset')
            if self.action_type == 'continuous':
                self.mu, self.sigma = Nn.actor_continuous('actor', self.s, self.a_counts, reuse=False)
                self.norm_dist = tf.distributions.Normal(loc=self.mu, scale=self.sigma + self.sigma_offset)
                self.sample_op = tf.clip_by_value(self.norm_dist.sample(), -1, 1)
                log_act_prob = self.norm_dist.log_prob(self.pl_a)
                self.pl_s_a = tf.concat((self.s, self.pl_a), axis=1)
                self.q = Nn.critic_q_one('critic', self.pl_s_a, reuse=False)
                self.next_mu, _ = Nn.actor_continuous('actor', self.s_, self.a_counts, reuse=True)
                self.s_next_mu = tf.concat((self.s_, self.next_mu), axis=1)
                self.max_q_next = tf.stop_gradient(Nn.critic_q_one('critic', self.s_next_mu, reuse=True))

                self.entropy = self.norm_dist.entropy()
                tf.summary.scalar('LOSS/entropy', tf.reduce_mean(self.entropy))
            else:
                # self.pl_a_hot = tf.squeeze(
                #     tf.one_hot(
                #         tf.matmul(
                #             self.pl_a,
                #             self.action_multiplication_factor[:, np.newaxis].astype(np.float32)
                #         ), self.a_counts), axis=1)
                self.pl_s_a_hot = tf.concat((self.s, self.pl_a), axis=1)
                self.action_probs = Nn.actor_discrete('actor', self.s, self.a_counts)
                self.q = Nn.critic_q_one('critic', self.pl_s_a_hot, reuse=False)
                self.sample_op = tf.argmax(self.action_probs, axis=1)
                log_act_prob = tf.log(tf.reduce_sum(tf.multiply(self.action_probs, self.pl_a), axis=1))[:, np.newaxis]
                self.s_a_all = tf.concat(
                    (tf.tile(self.s_, [self.a_counts, 1]), tf.one_hot([i for i in range(self.a_counts)], self.a_counts)),
                    axis=1)
                self.max_q_next = tf.stop_gradient(tf.reduce_max(
                    Nn.critic_q_one('critic', self.s_a_all, reuse=True),
                    axis=0, keepdims=True))

            self.action = tf.identity(self.sample_op, name='action')

            self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
            self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

            self.q_value = tf.stop_gradient(self.q)
            self.actor_loss = tf.reduce_mean(log_act_prob * self.q_value)
            self.critic_loss = tf.reduce_mean(tf.squared_difference(self.q, self.pl_r + self.gamma * (1 - self.pl_done) * self.max_q_next))
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_critic = optimizer.minimize(self.critic_loss, var_list=critic_vars + self.conv_vars)
            with tf.control_dependencies([self.train_critic]):
                self.train_actor = optimizer.minimize(-self.actor_loss, var_list=actor_vars + self.conv_vars, global_step=self.global_step)
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
　　　　　　　ｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　
　　　　　　ｘｘｘ　　　　　　　　　　ｘｘｘ　　ｘｘ　　　　
　　　　　　ｘｘｘ　　　　　　　　　　ｘｘ　　　　ｘｘ　　　
　　　　　　ｘ　ｘｘ　　　　　　　　　ｘｘ　　　　　　　　　
　　　　　ｘｘ　ｘｘ　　　　　　　　ｘｘｘ　　　　　　　　　
　　　　　ｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　　　　　　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘ　　　　ｘｘ　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘｘ　　ｘｘｘ　　　
　　　ｘｘｘ　　ｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘ　　　　　　　　
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
        self.off_store(s, a, r[:, np.newaxis], s_, done[:, np.newaxis])

    def learn(self, episode):
        s, a, r, s_, done = self.data.sample()
        pl_visual_s, pl_s = self.get_visual_and_vector_input(s)
        pl_visual_s_, pl_s_ = self.get_visual_and_vector_input(s_)
        summaries, _ = self.sess.run([self.summaries, self.train_sequence], feed_dict={
            self.pl_visual_s: pl_visual_s,
            self.pl_s: pl_s,
            self.pl_a: a if self.action_type == 'continuous' else sth.action_index2one_hot(a, self.a_dim_or_list),
            self.pl_r: r,
            self.pl_visual_s_: pl_visual_s_,
            self.pl_s_: pl_s_,
            self.pl_done: done,
            self.episode: episode,
            self.sigma_offset: np.full(self.a_counts, 0.01)
        })
        self.recorder.writer.add_summary(summaries, self.sess.run(self.global_step))
