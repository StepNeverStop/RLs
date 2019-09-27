import numpy as np
import tensorflow as tf
import Nn
from .policy import Policy


class DDPG(Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 action_type,
                 gamma=0.99,
                 ployak=0.995,
                 lr=5.0e-4,
                 max_episode=50000,
                 batch_size=100,
                 buffer_size=10000,
                 base_dir=None,
                 logger2file=False,
                 out_graph=False):
        assert action_type == 'continuous', 'ddpg only support continuous action space'
        super().__init__(s_dim, visual_sources, visual_resolution, a_dim_or_list, action_type, gamma, max_episode, base_dir, 'OFF', batch_size, buffer_size)
        self.ployak = ployak
        with self.graph.as_default():
            self.lr = tf.train.polynomial_decay(lr, self.episode, self.max_episode, 1e-10, power=1.0)
            # self.action_noise = Nn.NormalActionNoise(mu=np.zeros(self.a_counts), sigma=1 * np.ones(self.a_counts))
            self.action_noise = Nn.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_counts), sigma=0.2 * np.ones(self.a_counts))
            self.mu = Nn.actor_dpg('actor', self.pl_s, self.pl_visual_s, self.a_counts)
            tf.identity(self.mu, 'action')
            self.action = tf.clip_by_value(self.mu + self.action_noise(), -1, 1)

            self.target_mu = Nn.actor_dpg('actor_target', self.pl_s_, self.pl_visual_s_, self.a_counts)
            self.action_target = tf.clip_by_value(self.target_mu + self.action_noise(), -1, 1)

            self.q = Nn.critic_q_one('q', self.pl_s, self.pl_visual_s, self.pl_a)
            self.q_actor = Nn.critic_q_one('q', self.pl_s, self.pl_visual_s, self.mu)
            self.q_target = Nn.critic_q_one('q_target', self.pl_s_, self.pl_visual_s_, self.action_target) # must be action_target, not target_mu
            self.dc_r = tf.stop_gradient(self.pl_r + self.gamma * self.q_target * (1 - self.pl_done))

            self.q_loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.q, self.dc_r))
            self.actor_loss = -tf.reduce_mean(self.q_actor)

            self.q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q')
            self.q_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q_target')
            self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
            self.actor_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_target')

            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_q = optimizer.minimize(
                self.q_loss, var_list=self.q_vars, global_step=self.global_step)
            with tf.control_dependencies([self.train_q]):
                self.train_actor = optimizer.minimize(self.actor_loss, var_list=self.actor_vars)
            with tf.control_dependencies([self.train_actor]):
                self.assign_q_target = tf.group([tf.assign(r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.q_target_vars, self.q_vars)])
                self.assign_actor_target = tf.group([tf.assign(r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.actor_target_vars, self.actor_vars)])
            # self.assign_q_target = [tf.assign(r, 1/(self.episode+1) * v + (1-1/(self.episode+1)) * r) for r, v in zip(self.q_target_vars, self.q_vars)]
            # self.assign_q_target = [tf.assign(r, 1/(self.episode+1) * v + (1-1/(self.episode+1)) * r) for r, v in zip(self.actor_target_vars, self.actor_vars)]
            self.train_sequence = [self.train_q, self.train_actor, self.assign_q_target, self.assign_actor_target]

            tf.summary.scalar('LOSS/actor_loss', tf.reduce_mean(self.actor_loss))
            tf.summary.scalar('LOSS/critic_loss', tf.reduce_mean(self.q_loss))
            tf.summary.scalar('LEARNING_RATE/lr', tf.reduce_mean(self.lr))
            self.summaries = tf.summary.merge_all()
            self.generate_recorder(
                logger2file=logger2file,
                graph=self.graph if out_graph else None
            )
            self.recorder.logger.info('''
　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　　　　　　　
　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘ　　　ｘｘｘｘｘ　　　
　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　ｘｘｘ　　　　
　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　
            ''')
            self.recorder.logger.info(self.action_noise)

    def choose_action(self, s, visual_s):
        return self.sess.run(self.action, feed_dict={
            self.pl_visual_s: visual_s,
            self.pl_s: s,
        })

    def choose_inference_action(self, s, visual_s):
        return self.sess.run(self.mu, feed_dict={
            self.pl_visual_s: visual_s,
            self.pl_s: s,
        })

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        self.off_store(s, visual_s, a, r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis])

    def learn(self, episode):
        s, visual_s, a, r, s_, visual_s_, done = self.data.sample()
        summaries, _ = self.sess.run([self.summaries, self.train_sequence], feed_dict={
            self.pl_visual_s: visual_s,
            self.pl_s: s,
            self.pl_a: a,
            self.pl_r: r,
            self.pl_visual_s_: visual_s_,
            self.pl_s_: s_,
            self.pl_done: done,
            self.episode: episode
        })
        self.recorder.writer.add_summary(summaries, self.sess.run(self.global_step))
