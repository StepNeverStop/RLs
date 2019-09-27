import numpy as np
import tensorflow as tf
import Nn
from .policy import Policy


class SAC(Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 action_type,
                 alpha=0.2,
                 auto_adaption=True,
                 gamma=0.99,
                 ployak=0.995,
                 lr=5.0e-4,
                 max_episode=50000,
                 batch_size=100,
                 buffer_size=10000,
                 base_dir=None,
                 logger2file=False,
                 out_graph=False):
        assert action_type == 'continuous', 'sac only support continuous action space'
        super().__init__(s_dim, visual_sources, visual_resolution, a_dim_or_list, action_type, gamma, max_episode, base_dir, 'OFF', batch_size, buffer_size)
        self.ployak = ployak
        with self.graph.as_default():
            self.sigma_offset = tf.placeholder(tf.float32, [self.a_counts, ], 'sigma_offset')
            self.log_alpha = tf.get_variable('log_alpha', dtype=tf.float32, initializer=0.0)
            self.alpha = alpha if not auto_adaption else tf.exp(self.log_alpha)
            self.lr = tf.train.polynomial_decay(lr, self.episode, self.max_episode, 1e-10, power=1.0)

            self.mu, self.sigma = Nn.actor_continuous('actor_net', self.pl_s, self.pl_visual_s, self.a_counts)
            tf.identity(self.mu, 'action')
            self.norm_dist = tf.distributions.Normal(loc=self.mu, scale=self.sigma + self.sigma_offset)
            self.a_new = tf.clip_by_value(self.norm_dist.sample(), -1, 1)
            self.log_prob = self.norm_dist.log_prob(self.a_new)
            self.entropy = self.norm_dist.entropy()

            self.q1 = Nn.critic_q_one('q1', self.pl_s, self.pl_visual_s, self.pl_a)
            self.q2 = Nn.critic_q_one('q2', self.pl_s, self.pl_visual_s, self.pl_a)
            self.q1_anew = Nn.critic_q_one('q1', self.pl_s, self.pl_visual_s, self.a_new)
            self.q2_anew = Nn.critic_q_one('q2', self.pl_s, self.pl_visual_s, self.a_new)
            self.v_from_q_stop = tf.stop_gradient(tf.minimum(self.q1_anew, self.q2_anew) - self.alpha * self.log_prob)
            self.v = Nn.critic_v('v', self.pl_s, self.pl_visual_s)
            self.v_target = Nn.critic_v('v_target', self.pl_s_, self.pl_visual_s_)
            self.dc_r = tf.stop_gradient(self.pl_r + self.gamma * self.v_target * (1 - self.pl_done))

            self.q1_loss = tf.reduce_mean(tf.squared_difference(self.q1, self.dc_r))
            self.q2_loss = tf.reduce_mean(tf.squared_difference(self.q2, self.dc_r))
            self.v_loss_stop = tf.reduce_mean(tf.squared_difference(self.v, self.v_from_q_stop))
            self.critic_loss = 0.5 * self.q1_loss + 0.5 * self.q2_loss + 0.5 * self.v_loss_stop
            self.actor_loss = -tf.reduce_mean(self.q1_anew - self.alpha * self.log_prob)
            self.alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(self.log_prob - self.a_counts))

            self.q1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q1')
            self.q2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q2')
            self.v_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='v')
            self.v_target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='v_target')
            self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_net')

            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_q1 = optimizer.minimize(self.q1_loss, var_list=self.q1_vars)
            self.train_q2 = optimizer.minimize(self.q2_loss, var_list=self.q2_vars)
            self.train_v = optimizer.minimize(self.v_loss_stop, var_list=self.v_vars)

            self.assign_v_target = tf.group([tf.assign(r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.v_target_vars, self.v_vars)])
            # self.assign_v_target = [tf.assign(r, 1/(self.episode+1) * v + (1-1/(self.episode+1)) * r) for r, v in zip(self.v_target_vars, self.v_vars)]
            with tf.control_dependencies([self.assign_v_target]):
                self.train_critic = optimizer.minimize(self.critic_loss, var_list=self.q1_vars + self.q2_vars + self.v_vars, global_step=self.global_step)
            with tf.control_dependencies([self.train_critic]):
                self.train_actor = optimizer.minimize(self.actor_loss, var_list=self.actor_vars)
            with tf.control_dependencies([self.train_actor]):
                self.train_alpha = optimizer.minimize(self.alpha_loss, var_list=[self.log_alpha])
            self.train_sequence = [self.assign_v_target, self.train_critic, self.train_actor, self.train_alpha]

            tf.summary.scalar('LOSS/actor_loss', tf.reduce_mean(self.actor_loss))
            tf.summary.scalar('LOSS/critic_loss', tf.reduce_mean(self.critic_loss))
            tf.summary.scalar('LOSS/alpha', self.alpha)
            tf.summary.scalar('LOSS/entropy', tf.reduce_mean(self.entropy))
            tf.summary.scalar('LEARNING_RATE/lr', tf.reduce_mean(self.lr))
            self.summaries = tf.summary.merge_all()
            self.generate_recorder(
                logger2file=logger2file,
                graph=self.graph if out_graph else None
            )
            self.recorder.logger.info('''
　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　　ｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　　　ｘｘｘ　　　　　　　　　　ｘｘｘ　　ｘｘ　　　　
　　　　ｘｘ　　　　ｘ　　　　　　　　　　ｘｘｘ　　　　　　　　　　ｘｘ　　　　ｘｘ　　　
　　　　ｘｘｘｘ　　　　　　　　　　　　　ｘ　ｘｘ　　　　　　　　　ｘｘ　　　　　　　　　
　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘｘ　ｘｘ　　　　　　　　ｘｘｘ　　　　　　　　　
　　　　　　　　ｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　　　　　　　　
　　　　ｘ　　　　ｘｘ　　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘ　　　　ｘｘ　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘｘ　　ｘｘｘ　　　
　　　　ｘｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　ｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　
            ''')

    def choose_action(self, s, visual_s):
        return self.sess.run(self.a_new, feed_dict={
            self.pl_visual_s: visual_s,
            self.pl_s: s,
            self.sigma_offset: np.full(self.a_counts, 0.01)
        })

    def choose_inference_action(self, s, visual_s):
        return self.sess.run(self.mu, feed_dict={
            self.pl_visual_s: visual_s,
            self.pl_s: s,
            self.sigma_offset: np.full(self.a_counts, 0.01)
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
            self.episode: episode,
            self.sigma_offset: np.full(self.a_counts, 0.01)
        })
        self.recorder.writer.add_summary(summaries, self.sess.run(self.global_step))
