import numpy as np
import tensorflow as tf
import Nn
from utils.sth import sth
from .policy import Policy


class AC(Policy):
    # off-policy actor-critic
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
                 buffer_size=10000,
                 base_dir=None,
                 logger2file=False,
                 out_graph=False):
        super().__init__(s_dim, visual_sources, visual_resolution, a_dim_or_list, action_type, gamma, max_episode, base_dir, 'OFF', batch_size, buffer_size)
        with self.graph.as_default():
            self.lr = tf.train.polynomial_decay(lr, self.episode, self.max_episode, 1e-10, power=1.0)
            self.sigma_offset = tf.placeholder(tf.float32, [self.a_counts, ], 'sigma_offset')
            self.old_prob = tf.placeholder(tf.float32, [None, 1], 'old_prob')
            if self.action_type == 'continuous':
                self.mu, self.sigma = Nn.actor_continuous('actor', self.pl_s, self.pl_visual_s, self.a_counts)
                self.norm_dist = tf.distributions.Normal(loc=self.mu, scale=self.sigma + self.sigma_offset)
                self.sample_op = tf.clip_by_value(self.norm_dist.sample(), -1, 1)
                self.prob = tf.reduce_mean(self.norm_dist.prob(self.pl_a), axis=1)[:, np.newaxis]
                log_act_prob = self.norm_dist.log_prob(self.pl_a)
                self.q = Nn.critic_q_one('critic', self.pl_s, self.pl_visual_s, self.pl_a)
                self.next_mu, _ = Nn.actor_continuous('actor', self.pl_s_, self.pl_visual_s_, self.a_counts)
                self.max_q_next = tf.stop_gradient(Nn.critic_q_one('critic', self.pl_s_, self.pl_visual_s_, self.next_mu))

                self.entropy = self.norm_dist.entropy()
                tf.summary.scalar('LOSS/entropy', tf.reduce_mean(self.entropy))
            else:
                self.action_probs = Nn.actor_discrete('actor', self.pl_s, self.pl_visual_s, self.a_counts)
                self.q = Nn.critic_q_one('critic', self.pl_s, self.pl_visual_s, self.pl_a)
                self.sample_op = tf.argmax(self.action_probs, axis=1)
                self.prob = tf.reduce_sum(tf.multiply(self.action_probs, self.pl_a), axis=1)[:, np.newaxis]
                log_act_prob = tf.log(self.prob)
                # self.next_mu = tf.one_hot(tf.argmax(Nn.actor_discrete('actor', self.pl_s_, self.pl_visual_s_, self.a_counts), axis=-1), self.a_counts)
                # self.max_q_next = tf.stop_gradient(Nn.critic_q_one('critic', self.pl_s_, self.pl_visual_s_, self.next_mu))
                self._all_a = tf.expand_dims(tf.one_hot([i for i in range(self.a_counts)], self.a_counts), 1)
                self.all_a = tf.reshape(tf.tile(self._all_a, [1, tf.shape(self.pl_a)[0], 1]), [-1, self.a_counts])
                self.max_q_next = tf.stop_gradient(tf.reduce_max(
                    Nn.critic_q_one('critic', tf.tile(self.pl_s_, [self.a_counts, 1]), tf.tile(self.pl_visual_s_, [self.a_counts, 1]), self.all_a),
                    axis=0, keepdims=True))

            self.action = tf.identity(self.sample_op, name='action')
            self.ratio = tf.stop_gradient(self.prob / self.old_prob)
            self.q_value = tf.stop_gradient(self.q)

            self.actor_loss = tf.reduce_mean(self.ratio * log_act_prob * self.q_value)
            self.critic_loss = tf.reduce_mean(tf.squared_difference(self.q, self.pl_r + self.gamma * (1 - self.pl_done) * self.max_q_next))

            self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
            self.critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_critic = optimizer.minimize(self.critic_loss, var_list=self.critic_vars)
            with tf.control_dependencies([self.train_critic]):
                self.train_actor = optimizer.minimize(-self.actor_loss, var_list=self.actor_vars, global_step=self.global_step)
            self.train_sequence = [self.train_critic, self.train_actor]

            tf.summary.scalar('LOSS/actor_loss', tf.reduce_mean(-self.actor_loss))
            tf.summary.scalar('LOSS/critic_loss', tf.reduce_mean(self.critic_loss))
            tf.summary.scalar('LEARNING_RATE/lr', tf.reduce_mean(self.lr))
            self.summaries = tf.summary.merge_all()
            self.generate_recorder(
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

    def choose_action(self, s, visual_s):
        if self.action_type == 'continuous':
            return self.sess.run(self.action, feed_dict={
                self.pl_visual_s: visual_s,
                self.pl_s: s,
                self.sigma_offset: np.full(self.a_counts, 0.01)
            })
        else:
            if np.random.uniform() < 0.2:
                a = np.random.randint(0, self.a_counts, len(s))
            else:
                a = self.sess.run(self.action, feed_dict={
                    self.pl_visual_s: visual_s,
                    self.pl_s: s
                })
            return sth.int2action_index(a, self.a_dim_or_list)

    def choose_inference_action(self, s, visual_s):
        if self.action_type == 'continuous':
            return self.sess.run(self.mu, feed_dict={
                self.pl_visual_s: visual_s,
                self.pl_s: s,
                self.sigma_offset: np.full(self.a_counts, 0.01)
            })
        else:
            a = self.sess.run(self.action, feed_dict={
                self.pl_visual_s: visual_s,
                self.pl_s: s,
                self.sigma_offset: np.full(self.a_counts, 0.01)
            })
            return sth.int2action_index(a, self.a_dim_or_list)

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        old_prob = self.sess.run(self.prob, feed_dict={
            self.pl_visual_s: visual_s,
            self.pl_s: s,
            self.pl_a: a if self.action_type == 'continuous' else sth.action_index2one_hot(a, self.a_dim_or_list),
            self.sigma_offset: np.full(self.a_counts, 0.01)
        })
        assert isinstance(a, np.ndarray), "store_data need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store_data need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store_data need done type is np.ndarray"
        self.data.add(s, visual_s, a, old_prob, r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis])

    def no_op_store(self, s, visual_s, a, r, s_, visual_s_, done):
        old_prob = np.ones_like(r)
        assert isinstance(a, np.ndarray), "store_data need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store_data need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store_data need done type is np.ndarray"
        if self.policy_mode == 'OFF':
            self.data.add(s, visual_s, a, old_prob[:, np.newaxis], r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis])

    def learn(self, episode):
        s, visual_s, a, old_prob, r, s_, visual_s_, done = self.data.sample()
        summaries, _ = self.sess.run([self.summaries, self.train_sequence], feed_dict={
            self.pl_visual_s: visual_s,
            self.pl_s: s,
            self.pl_a: a if self.action_type == 'continuous' else sth.action_index2one_hot(a, self.a_dim_or_list),
            self.old_prob: old_prob,
            self.pl_r: r,
            self.pl_visual_s_: visual_s_,
            self.pl_s_: s_,
            self.pl_done: done,
            self.episode: episode,
            self.sigma_offset: np.full(self.a_counts, 0.01)
        })
        self.recorder.writer.add_summary(summaries, self.sess.run(self.global_step))
