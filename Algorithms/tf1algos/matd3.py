import numpy as np
import tensorflow as tf
import Nn
from .base import Base


class MATD3(Base):
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
        assert action_type == 'continuous', 'matd3 only support continuous action space'
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

            self.mumu = tf.concat((self.q_actor_a_previous, self.mu, self.q_actor_a_later), axis=1)

            self.q1 = Nn.critic_q_one('q1', self.ss, None, self.aa)
            self.q1_actor = Nn.critic_q_one('q1', self.ss, None, self.mumu)
            self.q1_target = Nn.critic_q_one('q1_target', self.ss_, None, self.aa_)

            self.q2 = Nn.critic_q_one('q2', self.ss, None, self.aa)
            self.q2_target = Nn.critic_q_one('q2_target', self.ss_, None, self.aa_)

            self.q_target = tf.minimum(self.q1_target, self.q2_target)
            self.dc_r = tf.stop_gradient(self.pl_r + self.gamma * self.q_target)

            self.q1_loss = tf.reduce_mean(tf.squared_difference(self.q1, self.dc_r))
            self.q2_loss = tf.reduce_mean(tf.squared_difference(self.q2, self.dc_r))
            self.critic_loss = 0.5 * (self.q1_loss + self.q2_loss)
            self.actor_loss = -tf.reduce_mean(self.q1_actor)

            self.q1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q1')
            self.q1_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q1_target')
            self.q2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q2')
            self.q2_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q2_target')
            self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
            self.actor_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_target')

            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_value = optimizer.minimize(self.critic_loss, var_list=self.q1_vars + self.q2_vars)
            with tf.control_dependencies([self.train_value]):
                self.train_actor = optimizer.minimize(self.actor_loss, var_list=self.actor_vars, global_step=self.global_step)
            with tf.control_dependencies([self.train_actor]):
                self.assign_q1_target = tf.group([tf.assign(r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.q1_target_vars, self.q1_vars)])
                self.assign_q2_target = tf.group([tf.assign(r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.q2_target_vars, self.q2_vars)])
                self.assign_actor_target = tf.group([tf.assign(r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.actor_target_vars, self.actor_vars)])
            self.train_sequence = [self.train_value, self.train_actor, self.assign_q1_target, self.assign_q2_target, self.assign_actor_target]

            tf.summary.scalar('LOSS/actor_loss', tf.reduce_mean(self.actor_loss))
            tf.summary.scalar('LOSS/critic_loss', tf.reduce_mean(self.critic_loss))
            tf.summary.scalar('LEARNING_RATE/lr', tf.reduce_mean(self.lr))
            self.summaries = tf.summary.merge_all()
            self.generate_recorder(
                logger2file=logger2file,
                graph=self.graph if out_graph else None
            )
            self.recorder.logger.info('''
　　ｘｘｘｘ　　　　ｘｘｘ　　　　　　　　　ｘｘ　　　　　　　　　ｘｘｘｘｘｘｘｘｘ　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　　　　　
　　　ｘｘｘ　　　　ｘｘ　　　　　　　　　ｘｘｘ　　　　　　　　　ｘｘ　　ｘ　　ｘｘ　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　ｘｘ　　　　　
　　　　ｘｘｘ　　ｘｘｘ　　　　　　　　　ｘｘｘ　　　　　　　　　ｘｘ　　ｘ　　ｘｘ　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘｘ　ｘｘ　　　　　
　　　　ｘｘｘ　　ｘｘｘ　　　　　　　　　ｘ　ｘｘ　　　　　　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　　　ｘｘｘ　　　　　
　　　　ｘｘｘｘ　ｘ　ｘ　　　　　　　　ｘｘ　ｘｘ　　　　　　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　　ｘｘｘｘ　　　　　
　　　　ｘ　ｘｘｘｘ　ｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　　　　ｘｘｘ　　　　
　　　　ｘ　ｘｘｘ　　ｘ　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　
　　　　ｘ　　ｘｘ　　ｘ　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　　　　ｘ　　　　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　ｘｘｘ　　　　
　　ｘｘｘｘ　ｘｘｘｘｘｘ　　　　　ｘｘｘ　　ｘｘｘｘｘ　　　　　　　ｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　
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
        self.sess.run(self.train_value, feed_dict={
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
