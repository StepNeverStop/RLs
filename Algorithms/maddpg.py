import numpy as np
import tensorflow as tf
import Nn
from .base import Base


class MADDPG(Base):
    def __init__(self,
                 s_dim,
                 a_dim_or_list,
                 action_type,
                 gamma=0.99,
                 ployak=0.995,
                 lr=5.0e-4,
                 max_episode=50000,
                 n=1,
                 i=0,
                 base_dir=None,
                 logger2file=False,
                 out_graph=False):
        assert action_type == 'continuous', 'maddpg only support continuous action space'
        super().__init__(a_dim_or_list, action_type, base_dir)
        self.n = n
        self.i = i
        self.s_dim = s_dim
        self.a_dim_or_list = a_dim_or_list
        self.gamma = gamma
        self.max_episode = max_episode
        self.ployak = ployak

        with self.graph.as_default():
            self.q_actor_a_previous = tf.placeholder(tf.float32, [None, (self.a_counts) * self.i], 'q_actor_a_previous')
            self.q_actor_a_later = tf.placeholder(tf.float32, [None, (self.a_counts) * (self.n - self.i - 1)], 'q_actor_a_later')
            self.ss = tf.placeholder(tf.float32, [None, (self.s_dim) * self.n], 'ss')
            self.ss_ = tf.placeholder(tf.float32, [None, (self.s_dim) * self.n], 'ss_')
            self.aa = tf.placeholder(tf.float32, [None, (self.a_counts) * self.n], 'aa')
            self.aa_ = tf.placeholder(tf.float32, [None, (self.a_counts) * self.n], 'aa_')
            self.pl_s = tf.placeholder(tf.float32, [None, self.s_dim], 'vector_observation')
            self.pl_r = tf.placeholder(tf.float32, [None, 1], 'reward')

            self.lr = tf.train.polynomial_decay(lr, self.episode, self.max_episode, 1e-10, power=1.0)
            # self.action_noise = Nn.NormalActionNoise(mu=np.zeros(self.a_counts), sigma=1 * np.ones(self.a_counts))
            self.action_noise = Nn.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_counts), sigma=0.2 * np.ones(self.a_counts))
            self.mu = Nn.actor_dpg('actor', self.pl_s, None, self.a_counts)
            tf.identity(self.mu, 'action')
            self.action = tf.clip_by_value(self.mu + self.action_noise(), -1, 1)

            self.target_mu = Nn.actor_dpg('actor_target', self.pl_s, None, self.a_counts)
            self.action_target = tf.clip_by_value(self.target_mu + self.action_noise(), -1, 1)

            self.mumu = tf.concat((self.q_actor_a_previous, self.mu, self.q_actor_a_later), axis=-1)

            self.q = Nn.critic_q_one('q', self.ss, None, self.aa)
            self.q_actor = Nn.critic_q_one('q', self.ss, None, self.mumu)
            self.q_target = Nn.critic_q_one('q_target', self.ss_, None, self.aa_)
            self.dc_r = tf.stop_gradient(self.pl_r + self.gamma * self.q_target)

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
　　ｘｘｘｘ　　　　ｘｘｘ　　　　　　　　　ｘｘ　　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　ｘｘｘ　　　　ｘｘ　　　　　　　　　ｘｘｘ　　　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　　　ｘｘｘ　　ｘｘｘ　　　　　　　　　ｘｘｘ　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　ｘｘｘ　　ｘｘｘ　　　　　　　　　ｘ　ｘｘ　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　　　　　　　
　　　　ｘｘｘｘ　ｘ　ｘ　　　　　　　　ｘｘ　ｘｘ　　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘ　　　ｘｘｘｘｘ　　　
　　　　ｘ　ｘｘｘｘ　ｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　ｘｘｘ　　　　
　　　　ｘ　ｘｘｘ　　ｘ　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　ｘ　　ｘｘ　　ｘ　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　ｘｘｘｘ　ｘｘｘｘｘｘ　　　　　ｘｘｘ　　ｘｘｘｘｘ　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　
            ''')
            self.recorder.logger.info(self.action_noise)

    def choose_action(self, s):
        return self.sess.run(self.action, feed_dict={
            self.pl_s: s,
        })

    def get_target_action(self, s):
        return self.sess.run(self.action_target, feed_dict={
            self.pl_s: s,
        })

    def choose_inference_action(self, s):
        return self.sess.run(self.mu, feed_dict={
            self.pl_s: s,
        })

    def learn(self, episode, ap, al, ss, ss_, aa, aa_, s, r):
        summaries, _ = self.sess.run([self.summaries, self.train_sequence], feed_dict={
            self.q_actor_a_previous: ap,
            self.q_actor_a_later: al,
            self.ss: ss,
            self.ss_: ss_,
            self.aa: aa,
            self.aa_: aa_,
            self.pl_s: s,
            self.pl_r: r,
            self.episode: episode
        })
        self.recorder.writer.add_summary(summaries, self.sess.run(self.global_step))

    def get_max_episode(self):
        """
        get the max episode of this training model.
        """
        return self.max_episode
