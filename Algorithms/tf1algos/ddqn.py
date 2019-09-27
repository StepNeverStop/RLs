import numpy as np
import tensorflow as tf
import Nn
from utils.sth import sth
from .policy import Policy


class DDQN(Policy):
    '''
    Double DQN
    '''

    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 action_type,
                 lr=5.0e-4,
                 gamma=0.99,
                 epsilon=0.2,
                 max_episode=50000,
                 batch_size=100,
                 buffer_size=10000,
                 assign_interval=2,
                 base_dir=None,
                 logger2file=False,
                 out_graph=False):
        assert action_type == 'discrete', 'double dqn only support discrete action space'
        super().__init__(s_dim, visual_sources, visual_resolution, a_dim_or_list, action_type, gamma, max_episode, base_dir, 'OFF', batch_size=batch_size, buffer_size=buffer_size)
        self.epsilon = epsilon
        self.assign_interval = assign_interval
        with self.graph.as_default():
            self.lr = tf.train.polynomial_decay(lr, self.episode, self.max_episode, 1e-10, power=1.0)

            self.q = Nn.critic_q_all('q', self.pl_s, self.pl_visual_s, self.a_counts)
            self.action = tf.argmax(self.q, axis=1)
            tf.identity(self.action, 'action')

            self.q_next = Nn.critic_q_all('q', self.pl_s_, self.pl_visual_s_, self.a_counts)
            self.next_max_action = tf.argmax(self.q_next, axis=1)
            self.next_max_action_one_hot = tf.one_hot(tf.squeeze(self.next_max_action), self.a_counts, 1., 0., dtype=tf.float32)
            self.q_target_next = Nn.critic_q_all('q_target', self.pl_s_, self.pl_visual_s_, self.a_counts)

            self.q_eval = tf.reduce_sum(tf.multiply(self.q, self.pl_a), axis=1)[:, np.newaxis]
            self.q_target_next_max = tf.reduce_sum(tf.multiply(self.q_target_next, self.next_max_action_one_hot), axis=1)[:, np.newaxis]
            self.q_target = tf.stop_gradient(self.pl_r + self.gamma * (1 - self.pl_done) * self.q_target_next_max)

            self.q_loss = tf.reduce_mean(tf.squared_difference(self.q_eval, self.q_target))

            self.q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q')
            self.q_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_target')

            self.train_q = tf.train.AdamOptimizer(self.lr).minimize(self.q_loss, var_list=self.q_vars, global_step=self.global_step)
            self.assign_q_target = tf.group([tf.assign(r, v) for r, v in zip(self.q_target_vars, self.q_vars)])

            tf.summary.scalar('LOSS/loss', tf.reduce_mean(self.q_loss))
            tf.summary.scalar('LEARNING_RATE/lr', tf.reduce_mean(self.lr))
            self.summaries = tf.summary.merge_all()
            self.generate_recorder(
                logger2file=logger2file,
                graph=self.graph if out_graph else None
            )
            self.recorder.logger.info('''
　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　ｘｘｘｘ　　　ｘｘｘｘ　　
　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘ　ｘｘｘｘ　　　　　　　ｘｘｘ　　　　ｘ　　　
　　　　ｘｘ　　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　ｘｘｘｘ　　　　　　ｘｘｘｘ　　　ｘ　　　
　　　　ｘｘ　　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘｘｘｘｘ　　ｘ　　　
　　　　ｘｘ　　　　　ｘｘ　　　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘｘ　　　　　　ｘ　ｘｘｘｘ　ｘ　　　
　　　　ｘｘ　　　　　ｘｘ　　　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘ　　ｘｘｘｘｘ　　　
　　　　ｘｘ　　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘ　　　ｘｘｘｘ　　　
　　　　ｘｘ　　　ｘｘｘｘ　　　　　　ｘｘ　　　ｘｘｘｘ　　　　　ｘｘｘ　　　ｘｘｘ　　　　　　　ｘ　　　　ｘｘｘ　　　
　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　ｘｘｘ　　　　ｘｘ　　　
　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　　　　　　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘ　　　　　　　　　　　　　　　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘ　　　　　　　　　　　　　　　　
            ''')

    def choose_action(self, s, visual_s):
        if np.random.uniform() < self.epsilon:
            a = np.random.randint(0, self.a_counts, len(s))
        else:
            a = self.sess.run(self.action, feed_dict={
                self.pl_visual_s: visual_s,
                self.pl_s: s
            })
        return sth.int2action_index(a, self.a_dim_or_list)

    def choose_inference_action(self, s, visual_s):
        return sth.int2action_index(
            self.sess.run(self.action, feed_dict={
                self.pl_visual_s: visual_s,
                self.pl_s: s
            }),
            self.a_dim_or_list
        )

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        self.off_store(s, visual_s, a, r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis])

    def learn(self, episode):
        s, visual_s, a, r, s_, visual_s_, done = self.data.sample()
        _a = sth.action_index2one_hot(a, self.a_dim_or_list)
        summaries, _ = self.sess.run([self.summaries, self.train_q], feed_dict={
            self.pl_visual_s: visual_s,
            self.pl_s: s,
            self.pl_a: _a,
            self.pl_r: r,
            self.pl_visual_s_: visual_s_,
            self.pl_s_: s_,
            self.pl_done: done,
            self.episode: episode
        })
        if self.sess.run(self.global_step) % self.assign_interval == 0:
            self.sess.run(self.assign_q_target)
        self.recorder.writer.add_summary(summaries, self.sess.run(self.global_step))
