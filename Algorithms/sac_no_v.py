import numpy as np
import tensorflow as tf
import Nn
from Algorithms.algorithm_base import Policy


class SAC_NO_V(Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolutions,
                 a_dim_or_list,
                 action_type,
                 alpha=0.2,
                 auto_adaption=True,
                 lr=5.0e-4,
                 max_episode=50000,
                 gamma=0.99,
                 ployak=0.995,
                 batch_size=100,
                 buffer_size=10000,
                 cp_dir=None,
                 log_dir=None,
                 excel_dir=None,
                 logger2file=False,
                 out_graph=False):
        super().__init__(s_dim, visual_sources, visual_resolutions, a_dim_or_list, action_type, max_episode, cp_dir, 'OFF', batch_size, buffer_size)
        self.gamma = gamma
        self.ployak = ployak
        with self.graph.as_default():
            self.log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
            self.alpha = alpha if not auto_adaption else tf.exp(self.log_alpha)

            self.lr = tf.train.polynomial_decay(lr, self.episode, self.max_episode, 1e-10, power=1.0)
            self.sigma_offset = tf.placeholder(tf.float32, [self.a_counts, ], 'sigma_offset')

            self.mu, self.sigma = Nn.actor_continuous('actor_net', self.s, self.a_counts, reuse=False)
            self.norm_dist = tf.distributions.Normal(loc=self.mu, scale=self.sigma + self.sigma_offset)
            self.a_s = tf.clip_by_value(self.norm_dist.sample(), -1, 1)
            self.a_s_log_prob = self.norm_dist.log_prob(self.a_s)
            tf.identity(self.mu, 'action')
            target_mu, target_sigma = Nn.actor_continuous('actor_net', self.s_, self.a_counts, reuse=True)
            target_norm_dist = tf.distributions.Normal(loc=target_mu, scale=target_sigma + self.sigma_offset)
            self.a_s_ = tf.clip_by_value(target_norm_dist.sample(), -1, 1)
            self.a_s_log_prob_ = target_norm_dist.log_prob(self.a_s_)

            self.prob = self.norm_dist.prob(self.a_s)
            self.new_log_prob = self.norm_dist.log_prob(self.pl_a)
            self.entropy = self.norm_dist.entropy()
            self.s_a = tf.concat((self.s, self.pl_a), axis=1)
            self.s_a_ = tf.concat((self.s_, self.a_s_), axis=1)
            self.s_a_s = tf.concat((self.s, self.a_s), axis=1)
            self.q1 = Nn.critic_q_one('q1', self.s_a, trainable=True, reuse=False)
            self.q1_target = Nn.critic_q_one('q1_target', self.s_a_, trainable=False, reuse=False)
            self.q2 = Nn.critic_q_one('q2', self.s_a, trainable=True, reuse=False)
            self.q2_target = Nn.critic_q_one('q2_target', self.s_a_, trainable=False, reuse=False)
            self.q1_s_a = Nn.critic_q_one('q1', self.s_a_s, trainable=True, reuse=True)
            self.q2_s_a = Nn.critic_q_one('q2', self.s_a_s, trainable=True, reuse=True)

            self.dc_r_q1 = tf.stop_gradient(self.pl_r + self.gamma * (self.q1_target - self.alpha * tf.reduce_mean(self.a_s_log_prob_)))
            self.dc_r_q2 = tf.stop_gradient(self.pl_r + self.gamma * (self.q2_target - self.alpha * tf.reduce_mean(self.a_s_log_prob_)))
            self.q1_loss = tf.reduce_mean(tf.squared_difference(self.q1, self.dc_r_q1))
            self.q2_loss = tf.reduce_mean(tf.squared_difference(self.q2, self.dc_r_q2))
            self.critic_loss = 0.5 * self.q1_loss + 0.5 * self.q2_loss
            # self.actor_loss = -tf.reduce_mean(tf.minimum(self.q1_s_a, self.q2_s_a) - self.alpha * (self.a_s_log_prob + self.new_log_prob))
            self.actor_loss = -tf.reduce_mean(tf.minimum(self.q1_s_a, self.q2_s_a) - self.alpha * self.a_s_log_prob)

            self.alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(self.a_s_log_prob - self.a_counts))

            self.q1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q1')
            self.q1_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q1_target')
            self.q2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q2')
            self.q2_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q2_target')
            self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_net')

            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_q1 = optimizer.minimize(self.q1_loss, var_list=self.q1_vars)
            self.train_q2 = optimizer.minimize(self.q2_loss, var_list=self.q2_vars)

            self.assign_q1_target = tf.group([tf.assign(r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.q1_target_vars, self.q1_vars)])
            self.assign_q2_target = tf.group([tf.assign(r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.q2_target_vars, self.q2_vars)])
            with tf.control_dependencies([self.assign_q1_target, self.assign_q2_target]):
                self.train_critic = optimizer.minimize(self.critic_loss, var_list=self.q1_vars + self.q2_vars + self.conv_vars, global_step=self.global_step)
            with tf.control_dependencies([self.train_critic]):
                self.train_actor = optimizer.minimize(self.actor_loss, var_list=self.actor_vars + self.conv_vars)
            with tf.control_dependencies([self.train_actor]):
                self.train_alpha = optimizer.minimize(self.alpha_loss, var_list=[self.log_alpha])
            self.train_sequence = [self.assign_q1_target, self.assign_q2_target, self.train_critic, self.train_actor, self.train_alpha]

            tf.summary.scalar('LOSS/actor_loss', tf.reduce_mean(self.actor_loss))
            tf.summary.scalar('LOSS/critic_loss', tf.reduce_mean(self.critic_loss))
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
　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　　ｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘ　　ｘｘｘ　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　　　ｘｘｘ　　　　　　　　　　ｘｘｘ　　ｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘ　　　　ｘ　　　　
　　　　ｘｘ　　　　ｘ　　　　　　　　　　ｘｘｘ　　　　　　　　　　ｘｘ　　　　ｘｘ　　　　　　　ｘｘ　　　　　　　　　　　　　　　　ｘｘ　　　　　　　　　　ｘｘ　　　ｘｘ　　　　
　　　　ｘｘｘｘ　　　　　　　　　　　　　ｘ　ｘｘ　　　　　　　　　ｘｘ　　　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　ｘｘｘ　　ｘｘ　　　　
　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘｘ　ｘｘ　　　　　　　　ｘｘｘ　　　　　　　　　　　　　　ｘｘ　ｘｘ　　　　　　　　　　ｘｘ　　ｘｘ　　　　　　　　　ｘｘ　ｘｘ　　　　　
　　　　　　　　ｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　　　　　　　　　　　　　ｘ　　　ｘ　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘｘｘｘｘ　　　　　
　　　　ｘ　　　　ｘｘ　　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘ　　　　ｘｘ　　　　　　　　ｘ　　　ｘ　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　　ｘｘｘ　　　　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘｘ　　ｘｘｘ　　　　　　　　ｘ　　　ｘ　　　　　　　　　　ｘｘ　　ｘｘ　　　　　　　　　　ｘｘｘ　　　　　　
　　　　ｘｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　ｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘ　　　　　　　　ｘｘｘ　ｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　　ｘ　　　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　　　　　　　　　　　ｘ　　　
            ''')
            self.init_or_restore(cp_dir)

    def choose_action(self, s):
        pl_visual_s, pl_s = self.get_visual_and_vector_input(s)
        return self.sess.run(self.a_s, feed_dict={
            self.pl_visual_s: pl_visual_s,
            self.pl_s: pl_s,
            self.sigma_offset: np.full(self.a_counts, 0.01)
        })

    def choose_inference_action(self, s):
        pl_visual_s, pl_s = self.get_visual_and_vector_input(s)
        return self.sess.run(self.mu, feed_dict={
            self.pl_visual_s: pl_visual_s,
            self.pl_s: pl_s,
            self.sigma_offset: np.full(self.a_counts, 0.01)
        })

    def store_data(self, s, a, r, s_, done):
        self.off_store(s, a, r[:, np.newaxis], s_, done[:, np.newaxis])

    def learn(self, episode):
        s, a, r, s_, _ = self.data.sample()
        pl_visual_s, pl_s = self.get_visual_and_vector_input(s)
        pl_visual_s_, pl_s_ = self.get_visual_and_vector_input(s_)
        # self.sess.run([self.train_q1, self.train_q2, self.train_actor, self.train_alpha, self.assign_q1_target, self.assign_q2_target], feed_dict={
        #     self.pl_visual_s: pl_visual_s,
        #     self.pl_s: pl_s,
        #     self.pl_a: a,
        #     self.pl_r: r,
        #     self.pl_visual_s_: pl_visual_s_,
        #     self.pl_s_: pl_s_,
        #     self.episode: episode,
        #     self.sigma_offset: np.full(self.a_counts, 0.01)
        # })
        summaries, _ = self.sess.run([self.summaries, self.train_sequence], feed_dict={
            self.pl_visual_s: pl_visual_s,
            self.pl_s: pl_s,
            self.pl_a: a,
            self.pl_r: r,
            self.pl_visual_s_: pl_visual_s_,
            self.pl_s_: pl_s_,
            self.episode: episode,
            self.sigma_offset: np.full(self.a_counts, 0.01)
        })
        self.recorder.writer.add_summary(summaries, self.sess.run(self.global_step))
