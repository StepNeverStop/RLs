import numpy as np
import tensorflow as tf
import Nn
from .policy import Policy


class TD3(Policy):
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
        assert action_type == 'continuous', 'td3 only support continuous action space'
        super().__init__(s_dim, visual_sources, visual_resolution, a_dim_or_list, action_type, gamma, max_episode, base_dir, 'OFF', batch_size, buffer_size)
        self.ployak = ployak
        with self.graph.as_default():
            self.lr = tf.train.polynomial_decay(lr, self.episode, self.max_episode, 1e-10, power=1.0)

            # self.action_noise = Nn.NormalActionNoise(mu=np.zeros(self.a_counts), sigma=1 * np.ones(self.a_counts))
            self.action_noise = Nn.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_counts), sigma=0.2 * np.ones(self.a_counts))
            self.mu = Nn.actor_dpg('actor', self.pl_s, self.pl_visual_s, self.a_counts)
            self.action = tf.clip_by_value(self.mu + self.action_noise(), -1, 1)
            tf.identity(self.mu, 'action')
            self.target_mu = Nn.actor_dpg('actor_target', self.pl_s_, self.pl_visual_s_, self.a_counts)
            self.action_target = tf.clip_by_value(self.target_mu + self.action_noise(), -1, 1)

            self.q1 = Nn.critic_q_one('q1', self.pl_s, self.pl_visual_s, self.pl_a)
            self.q1_actor = Nn.critic_q_one('q1', self.pl_s, self.pl_visual_s, self.mu)
            self.q1_target = Nn.critic_q_one('q1_target', self.pl_s_, self.pl_visual_s_, self.action_target)

            self.q2 = Nn.critic_q_one('q2', self.pl_s, self.pl_visual_s, self.pl_a)
            self.q2_target = Nn.critic_q_one('q2_target', self.pl_s_, self.pl_visual_s_, self.action_target)

            self.q_target = tf.minimum(self.q1_target, self.q2_target)
            self.dc_r = tf.stop_gradient(self.pl_r + self.gamma * self.q_target * (1 - self.pl_done))

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
            self.train_q1 = optimizer.minimize(self.q1_loss, var_list=self.q1_vars)
            self.train_q2 = optimizer.minimize(self.q2_loss, var_list=self.q2_vars)
            self.train_value = optimizer.minimize(self.critic_loss, var_list=self.q1_vars + self.q2_vars)
            with tf.control_dependencies([self.train_value]):
                self.train_actor = optimizer.minimize(self.actor_loss, var_list=self.actor_vars, global_step=self.global_step)
            with tf.control_dependencies([self.train_actor]):
                self.assign_q1_target = tf.group([tf.assign(r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.q1_target_vars, self.q1_vars)])
                self.assign_q2_target = tf.group([tf.assign(r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.q2_target_vars, self.q2_vars)])
                self.assign_actor_target = tf.group([tf.assign(r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.actor_target_vars, self.actor_vars)])
            self.train_sequence = [self.train_value, self.train_actor, self.assign_q1_target, self.assign_q2_target, self.assign_actor_target]
            # self.assign_q1_target = [
            #     tf.assign(r, 1/(self.episode+1) * v + (1-1/(self.episode+1)) * r) for r, v in zip(self.q1_target_vars, self.q1_vars)]
            # self.assign_q2_target = [
            #     tf.assign(r, 1/(self.episode+1) * v + (1-1/(self.episode+1)) * r) for r, v in zip(self.q2_target_vars, self.q2_vars)]
            # self.assign_actor_target = [
            #     tf.assign(r, 1/(self.episode+1) * v + (1-1/(self.episode+1)) * r) for r, v in zip(self.actor_target_vars, self.actor_vars)]

            tf.summary.scalar('LOSS/actor_loss', tf.reduce_mean(self.actor_loss))
            tf.summary.scalar('LOSS/critic_loss', tf.reduce_mean(self.critic_loss))
            tf.summary.scalar('LEARNING_RATE/lr', tf.reduce_mean(self.lr))
            self.summaries = tf.summary.merge_all()
            self.generate_recorder(
                logger2file=logger2file,
                graph=self.graph if out_graph else None
            )
            self.recorder.logger.info('''
　　　ｘｘｘｘｘｘｘｘｘ　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　　　　　
　　　ｘｘ　　ｘ　　ｘｘ　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　ｘｘ　　　　　
　　　ｘｘ　　ｘ　　ｘｘ　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘｘ　ｘｘ　　　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　　　ｘｘｘ　　　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　　ｘｘｘｘ　　　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　　　　ｘｘｘ　　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　
　　　　　　　ｘ　　　　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　ｘｘｘ　　　　
　　　　　ｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　
            ''')

    def choose_action(self, s, visual_s):
        return self.sess.run(self.action, feed_dict={
            self.pl_visual_s: visual_s,
            self.pl_s: s
        })

    def choose_inference_action(self, s, visual_s):
        return self.sess.run(self.mu, feed_dict={
            self.pl_visual_s: visual_s,
            self.pl_s: s
        })

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        self.off_store(s, visual_s, a, r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis])

    def learn(self, episode):
        s, visual_s, a, r, s_, visual_s_, done = self.data.sample()
        self.sess.run(self.train_value, feed_dict={
            self.pl_visual_s: visual_s,
            self.pl_s: s,
            self.pl_a: a,
            self.pl_r: r,
            self.pl_visual_s_: visual_s_,
            self.pl_s_: s_,
            self.pl_done: done,
            self.episode: episode
        })
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
