import numpy as np
import tensorflow as tf
from Algorithms.algorithm_base import Policy

initKernelAndBias = {
    'kernel_initializer': tf.random_normal_initializer(0., .1),
    'bias_initializer': tf.constant_initializer(0.1, dtype=tf.float32)
}


class TD3(Policy):
    def __init__(self,
                 s_dim,
                 a_counts,
                 action_type,
                 gamma=0.99,
                 ployak=0.995,
                 lr=5.0e-4,
                 max_episode=50000,
                 batch_size=100,
                 buffer_size=10000,
                 cp_dir=None,
                 log_dir=None,
                 excel_dir=None,
                 logger2file=False,
                 out_graph=False):
        super().__init__(s_dim, a_counts, action_type, max_episode, cp_dir, 'OFF', batch_size, buffer_size)
        self.gamma = gamma
        self.ployak = ployak
        with self.graph.as_default():
            self.lr = tf.train.polynomial_decay(lr, self.episode, self.max_episode, 1e-10, power=1.0)
            self.r = tf.placeholder(tf.float32, [None, 1], 'reward')
            self.s_ = tf.placeholder(tf.float32, [None, self.s_dim], 'next_state')

            self.mu, self.action, self.actor_var = self._build_actor_net('actor', self.pl_s, trainable=True)
            tf.identity(self.mu, 'action')
            self.target_mu, self.action_target, self.actor_target_var = self._build_actor_net('actor_target', self.s_, trainable=False)

            self.s_a = tf.concat((self.pl_s, self.pl_a), axis=1)
            self.s_mu = tf.concat((self.pl_s, self.mu), axis=1)
            self.s_a_target = tf.concat((self.s_, self.action_target), axis=1)

            self.q1, self.q1_var = self._build_q_net('q1', self.s_a, True, reuse=False)
            self.q1_actor, _ = self._build_q_net('q1', self.s_mu, True, reuse=True)
            self.q1_target, self.q1_target_var = self._build_q_net('q1_target', self.s_a_target, False, reuse=False)

            self.q2, self.q2_var = self._build_q_net('q2', self.s_a, True, reuse=False)
            self.q2_target, self.q2_target_var = self._build_q_net('q2_target', self.s_a_target, False, reuse=False)

            self.q_target = tf.minimum(self.q1_target, self.q2_target)
            self.dc_r = tf.stop_gradient(self.r + self.gamma * self.q_target)

            self.q1_loss = tf.reduce_mean(tf.squared_difference(self.q1, self.dc_r))
            self.q2_loss = tf.reduce_mean(tf.squared_difference(self.q2, self.dc_r))
            self.critic_loss = 0.5 * self.q1_loss + 0.5 * self.q2_loss
            self.actor_loss = -tf.reduce_mean(self.q1_actor)

            q1_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q1')
            q2_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q2')
            actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_q1 = optimizer.minimize(self.q1_loss, var_list=q1_var)
            self.train_q2 = optimizer.minimize(self.q2_loss, var_list=q2_var)
            self.train_value = optimizer.minimize(self.critic_loss)
            with tf.control_dependencies([self.train_value]):
                self.train_actor = optimizer.minimize(self.actor_loss, var_list=actor_vars, global_step=self.global_step)
            with tf.control_dependencies([self.train_actor]):
                self.assign_q1_target = tf.group([tf.assign(r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.q1_target_var, self.q1_var)])
                self.assign_q2_target = tf.group([tf.assign(r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.q2_target_var, self.q2_var)])
                self.assign_actor_target = tf.group([tf.assign(r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.actor_target_var, self.actor_var)])
            # self.assign_q1_target = [
            #     tf.assign(r, 1/(self.episode+1) * v + (1-1/(self.episode+1)) * r) for r, v in zip(self.q1_target_var, self.q1_var)]
            # self.assign_q2_target = [
            #     tf.assign(r, 1/(self.episode+1) * v + (1-1/(self.episode+1)) * r) for r, v in zip(self.q2_target_var, self.q2_var)]
            # self.assign_actor_target = [
            #     tf.assign(r, 1/(self.episode+1) * v + (1-1/(self.episode+1)) * r) for r, v in zip(self.actor_target_var, self.actor_var)]
            tf.summary.scalar('LOSS/actor_loss', tf.reduce_mean(self.actor_loss))
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
            self.init_or_restore(cp_dir)

    def _build_actor_net(self, name, input_vector, trainable):
        with tf.variable_scope(name):
            actor1 = tf.layers.dense(
                inputs=input_vector,
                units=128,
                activation=self.activation_fn,
                name='actor1',
                trainable=trainable,
                **initKernelAndBias
            )
            actor2 = tf.layers.dense(
                inputs=actor1,
                units=64,
                activation=self.activation_fn,
                name='actor2',
                trainable=trainable,
                **initKernelAndBias
            )
            mu = tf.layers.dense(
                inputs=actor2,
                units=self.a_counts,
                activation=tf.nn.tanh,
                name='mu',
                trainable=trainable,
                **initKernelAndBias
            )
            e = tf.random_normal(tf.shape(mu))
            action = tf.clip_by_value(mu + e, -1, 1)
            var = tf.get_variable_scope().global_variables()
        return mu, action, var

    def _build_q_net(self, name, input_vector, trainable, reuse=False):
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
            q1 = tf.layers.dense(
                inputs=layer2,
                units=1,
                activation=None,
                name='q_value',
                trainable=trainable,
                reuse=reuse,
                **initKernelAndBias
            )
            var = tf.get_variable_scope().global_variables()
        return q1, var

    def choose_action(self, s):
        return self.sess.run(self.action, feed_dict={
            self.pl_s: s
        })

    def choose_inference_action(self, s):
        return self.sess.run(self.mu, feed_dict={
            self.pl_s: s
        })

    def store_data(self, s, a, r, s_, done):
        assert isinstance(s, np.ndarray)
        assert isinstance(a, np.ndarray)
        assert isinstance(r, np.ndarray)
        assert isinstance(s_, np.ndarray)
        assert isinstance(done, np.ndarray)

        self.off_store(s, a, r, s_, done)

    def learn(self, episode):
        s, a, r, s_, _ = self.data.sample()
        self.sess.run(self.train_value, feed_dict={
            self.pl_s: s,
            self.pl_a: a,
            self.r: r,
            self.s_: s_,
            self.episode: episode
        })
        summaries, _ = self.sess.run([self.summaries, [self.train_value, self.train_actor, self.assign_q1_target, self.assign_q2_target, self.assign_actor_target]], feed_dict={
            self.pl_s: s,
            self.pl_a: a,
            self.r: r,
            self.s_: s_,
            self.episode: episode
        })
        self.recorder.writer.add_summary(summaries, self.sess.run(self.global_step))
