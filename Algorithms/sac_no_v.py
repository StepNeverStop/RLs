import numpy as np
import tensorflow as tf
from Algorithms.algorithm_base import Policy


initKernelAndBias = {
    'kernel_initializer': tf.random_normal_initializer(0., .1),
    'bias_initializer': tf.constant_initializer(0.1, dtype=tf.float32)
}


class SAC_NO_V(Policy):
    def __init__(self,
                 s_dim,
                 a_counts,
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
        super().__init__(s_dim, a_counts, cp_dir, 'OFF', batch_size, buffer_size)
        self.gamma = gamma
        self.ployak = ployak
        with self.graph.as_default():
            self.log_alpha = tf.get_variable(
                'log_alpha', dtype=tf.float32, initializer=0.0)
            self.alpha = alpha if not auto_adaption else tf.exp(
                self.log_alpha)

            self.lr = tf.train.polynomial_decay(
                lr, self.global_step, max_episode, 1e-10, power=1.0)
            self.r = tf.placeholder(tf.float32, [None, 1], 'reward')
            self.s_ = tf.placeholder(
                tf.float32, [None, self.s_dim], 'next_state')
            self.sigma_offset = tf.placeholder(
                tf.float32, [self.a_counts, ], 'sigma_offset')

            self.norm_dist, self.mu, self.sigma, self.a_s, self.a_s_log_prob = self._build_actor_net(
                'actor_net', self.pl_s, reuse=False)
            tf.identity(self.mu, 'action')
            _, _, _, self.a_s_, self.a_s_log_prob_ = self._build_actor_net(
                'actor_net', self.s_, reuse=True)
            self.prob = self.norm_dist.prob(self.a_s)
            self.new_log_prob = self.norm_dist.log_prob(self.pl_a)
            self.entropy = self.norm_dist.entropy()
            self.s_a = tf.concat((self.pl_s, self.pl_a), axis=1)
            self.s_a_ = tf.concat((self.s_, self.a_s_), axis=1)
            self.s_a_s = tf.concat((self.pl_s, self.a_s), axis=1)
            self.q1, self.q1_vars = self._build_q_net(
                'q1', self.s_a, trainable=True, reuse=False)
            self.q1_target, self.q1_target_vars = self._build_q_net(
                'q1_target', self.s_a_, trainable=False, reuse=False)
            self.q2, self.q2_vars = self._build_q_net(
                'q2', self.s_a, trainable=True, reuse=False)
            self.q2_target, self.q2_target_vars = self._build_q_net(
                'q2_target', self.s_a_, trainable=False, reuse=False)
            self.q1_s_a, _ = self._build_q_net(
                'q1', self.s_a_s, trainable=True, reuse=True)
            self.q2_s_a, _ = self._build_q_net(
                'q2', self.s_a_s, trainable=True, reuse=True)

            self.dc_r_q1 = tf.stop_gradient(
                self.r + self.gamma * (self.q1_target - self.alpha * tf.reduce_mean(self.a_s_log_prob_)))
            self.dc_r_q2 = tf.stop_gradient(
                self.r + self.gamma * (self.q2_target - self.alpha * tf.reduce_mean(self.a_s_log_prob_)))
            self.q1_loss = tf.reduce_mean(
                tf.squared_difference(self.q1, self.dc_r_q1))
            self.q2_loss = tf.reduce_mean(
                tf.squared_difference(self.q2, self.dc_r_q2))
            self.critic_loss = 0.5 * self.q1_loss + 0.5 * self.q2_loss
            # self.actor_loss = -tf.reduce_mean(
            #     tf.minimum(self.q1_s_a, self.q2_s_a) - self.alpha * (self.a_s_log_prob + self.new_log_prob))
            self.actor_loss = -tf.reduce_mean(
                tf.minimum(self.q1_s_a, self.q2_s_a) - self.alpha * self.a_s_log_prob)

            self.alpha_loss = -tf.reduce_mean(
                self.log_alpha * tf.stop_gradient(self.a_s_log_prob - self.a_counts))

            q1_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='q1')
            q2_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='q2')
            actor_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_net')

            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_q1 = optimizer.minimize(self.q1_loss, var_list=q1_vars)
            self.train_q2 = optimizer.minimize(self.q2_loss, var_list=q2_vars)

            self.assign_q1_target = tf.group([tf.assign(
                r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.q1_target_vars, self.q1_vars)])
            self.assign_q2_target = tf.group([tf.assign(
                r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.q2_target_vars, self.q2_vars)])
            with tf.control_dependencies([self.assign_q1_target, self.assign_q2_target]):
                self.train_critic = optimizer.minimize(
                    self.critic_loss, global_step=self.global_step)
            with tf.control_dependencies([self.train_critic]):
                self.train_actor = optimizer.minimize(
                    self.actor_loss, var_list=actor_vars)
            with tf.control_dependencies([self.train_actor]):
                self.train_alpha = optimizer.minimize(
                    self.alpha_loss, var_list=[self.log_alpha])
            # self.assign_q1_target = [
            #     tf.assign(r, 1/(self.global_step+1) * v + (1-1/(self.global_step+1)) * r) for r, v in zip(self.v_target_var, self.v_var)]
            # self.assign_q2_target = [
            #     tf.assign(r, 1/(self.global_step+1) * v + (1-1/(self.global_step+1)) * r) for r, v in zip(self.v_target_var, self.v_var)]

            tf.summary.scalar('LOSS/actor_loss',
                              tf.reduce_mean(self.actor_loss))
            tf.summary.scalar('LOSS/critic_loss',
                              tf.reduce_mean(self.critic_loss))
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
            self.init_or_restore(cp_dir, self.sess)

    def _build_actor_net(self, name, input_vector, reuse=False):
        with tf.variable_scope(name):
            actor1 = tf.layers.dense(
                inputs=input_vector,
                units=128,
                activation=self.activation_fn,
                name='actor1',
                reuse=reuse,
                **initKernelAndBias
            )
            actor2 = tf.layers.dense(
                inputs=actor1,
                units=64,
                activation=self.activation_fn,
                name='actor2',
                reuse=reuse,
                **initKernelAndBias
            )
            mu = tf.layers.dense(
                inputs=actor2,
                units=self.a_counts,
                activation=tf.nn.tanh,
                name='mu',
                reuse=reuse,
                **initKernelAndBias
            )
            sigma1 = tf.layers.dense(
                inputs=actor1,
                units=64,
                activation=self.activation_fn,
                name='simga1',
                reuse=reuse,
                **initKernelAndBias
            )
            sigma = tf.layers.dense(
                inputs=sigma1,
                units=self.a_counts,
                activation=tf.nn.sigmoid,
                name='sigma',
                reuse=reuse,
                **initKernelAndBias
            )
            norm_dist = tf.distributions.Normal(
                loc=mu, scale=sigma + self.sigma_offset)
            # action = tf.tanh(norm_dist.sample())
            action = tf.clip_by_value(
                norm_dist.sample(), -1, 1)
            log_prob = norm_dist.log_prob(action)
        return norm_dist, mu, sigma, action, log_prob

    def _build_q_net(self, name, input_vector, trainable, reuse):
        with tf.variable_scope(name):
            layer1 = tf.layers.dense(
                inputs=input_vector,
                units=256,
                activation=self.activation_fn,
                name='layer1',
                trainable=trainable,
                reuse=reuse,
                **initKernelAndBias
            )
            layer2 = tf.layers.dense(
                inputs=layer1,
                units=256,
                activation=self.activation_fn,
                name='layer2',
                trainable=trainable,
                reuse=reuse,
                **initKernelAndBias
            )
            q = tf.layers.dense(
                inputs=layer2,
                units=1,
                activation=None,
                name='q_value',
                trainable=trainable,
                reuse=reuse,
                **initKernelAndBias
            )
            var = tf.get_variable_scope().global_variables()
        return q, var

    def choose_action(self, s):
        return self.sess.run(self.a_s, feed_dict={
            self.pl_s: s,
            self.sigma_offset: np.full(self.a_counts, 0.01)
        })

    def choose_inference_action(self, s):
        return self.sess.run(self.mu, feed_dict={
            self.pl_s: s,
            self.sigma_offset: np.full(self.a_counts, 0.01)
        })

    def store_data(self, s, a, r, s_, done):
        assert isinstance(s, np.ndarray)
        assert isinstance(a, np.ndarray)
        assert isinstance(r, np.ndarray)
        assert isinstance(s_, np.ndarray)
        assert isinstance(done, np.ndarray)

        self.off_store(s, a, r, s_, done)

    def learn(self):
        s, a, r, s_, _ = self.data.sample()
        # self.sess.run([self.train_q1, self.train_q2, self.train_actor, self.train_alpha, self.assign_q1_target, self.assign_q2_target], feed_dict={
        #     self.pl_s: s,
        #     self.pl_a: a,
        #     self.r: r,
        #     self.s_: s_,
        #     self.sigma_offset: np.full(self.a_counts, 0.01)
        # })
        summaries, _ = self.sess.run([self.summaries, [self.assign_q1_target, self.assign_q2_target, self.train_critic, self.train_actor, self.train_alpha]], feed_dict={
            self.pl_s: s,
            self.pl_a: a,
            self.r: r,
            self.s_: s_,
            self.sigma_offset: np.full(self.a_counts, 0.01)
        })
        self.recorder.writer.add_summary(
            summaries, self.sess.run(self.global_step))
