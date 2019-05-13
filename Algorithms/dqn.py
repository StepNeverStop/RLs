import numpy as np
import tensorflow as tf
from utils.sth import sth
from Algorithms.algorithm_base import Policy

initKernelAndBias = {
    'kernel_initializer': tf.random_normal_initializer(0., .1),
    'bias_initializer': tf.constant_initializer(0.1, dtype=tf.float32)
}

class DQN(Policy):
    def __init__(
        self,
        s_dim,
        a_counts,
        action_type,

        lr=5.0e-4,
        gamma=0.99,
        epsilon=0.2,
        max_episode=50000,
        batch_size=100,
        buffer_size=10000,
        assign_interval=2,

        cp_dir=None,
        log_dir=None,
        excel_dir=None,
        logger2file=False,
        out_graph=False
    )
    super().__init__(s_dim, a_counts, action_type, max_episode, cp_dir, 'OFF', batch_size=batch_size, buffer_size = buffer_size)
    self.gamma=gamma
    self.epsilon=epsilon
    self.assign_interval=assign_interval
    with self.graph.as_default():
        self.r = tf.placeholder(tf.float32, [None, 1], 'reward')
        self.s_ = tf.placeholder(tf.float32, [None, self.s_dim], 'next_state')
        self.done = tf.placeholder(tf.float32, [None, 1], 'done')
        self.lr = tf.train.polynomial_decay(lr, self.episode, self.max_episode, 1e-10, power=1.0)
        self.q, self.q_var = self._build_q_net('q', self.s, trainable=True)
        self.q_next, self.q_target_var = self._build_q_net('q_target', self.s_, trainable=False)
        self.action = tf.argmax(self.q, axis=1)
        tf.identity(self.action, 'action')
        self.q_eval=tf.reduce_sum(tf.multiply(self.q,self.a),axis=1)
        self.q_target = self.r + self.gamma * (1-self.done) * tf.reduce_max(self.q_next, axis=1)
        
        self.q_loss = tf.reduce_mean(tf.squared_difference(self.q_eval, self.q_target))
        q_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q')
        self.train_q = tf.train.AdamOptimizer(self.lr).minimize(self.q_loss,var_list=q_vars, global_step=self.global_step)
        self.assign_q_target = tf.group([tf.assign(r, v) for r, v in zip(self.q_target_var, self.q_var)])

        tf.summary.scalar('LOSS/loss', tf.reduce_mean(self.q_loss))
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
　　　ｘｘｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　ｘｘｘｘ　　　ｘｘｘｘ　　
　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘ　ｘｘｘｘ　　　　　　　ｘｘｘ　　　　ｘ　　　
　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　ｘｘｘｘ　　　　　　ｘｘｘｘ　　　ｘ　　　
　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘｘｘｘｘ　　ｘ　　　
　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘｘ　　　　　　ｘ　ｘｘｘｘ　ｘ　　　
　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘ　　ｘｘｘｘｘ　　　
　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘ　　　ｘｘｘｘ　　　
　　　　ｘｘ　　　ｘｘｘｘ　　　　　ｘｘｘ　　　ｘｘｘ　　　　　　　ｘ　　　　ｘｘｘ　　　
　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　ｘｘｘ　　　　ｘｘ　　　
　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　　　　　　　　　　
　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘ　　　　　　　　　　　　　　　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘ
        ''')
        self.init_or_restore(cp_dir)
    
    def _build_q_net(self, name, input_vector, trainable):
        with tf.variable_scope(name):
            layer1 = tf.layers.dense(
                inputs=input_vector,
                units=256,
                activation=self.activation_fn,
                name='layer1',
                trainable=trainable,
                **initKernelAndBias
            )
            layer2 = tf.layers.dense(
                inputs=layer1,
                units=256,
                activation=self.activation_fn,
                name='layer2',
                trainable=trainable,
                **initKernelAndBias
            )
            q = tf.layers.dense(
                inputs=layer2,
                units=self.a_counts,
                activation=None,
                name='value',
                trainable=trainable,
                **initKernelAndBias
            )
            var = tf.get_variable_scope().global_variables()
        return q, var
    
    def choose_action(self,s):
        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, self.a_counts, s.shape[0])
        else:
            return self.sess.run(self.action, feed_dict={
                self.pl_s:s
            })
    def choose_inference_action(self, s):
        return self.sess.run(self.action, feed_dict={
                self.pl_s:s
        })
    
    def store_data(self, s, a, r, s_, done):
        assert isinstance(s, np.ndarray)
        assert isinstance(a, np.ndarray)
        assert isinstance(r, np.ndarray)
        assert isinstance(s_, np.ndarray)
        assert isinstance(done, np.ndarray)
        _a= np.zeros(self.a_counts)
        _a[a]=1
        self.off_store(s, _a, r, s_, done)

    def learn(self, episode):
        s, a, r, s_, done = self.data.sample()
        summaries, _ = self.sess.run([self.summaries, self.train_q], feed_dict={
            self.pl_s: s,
            self.pl_a: a,
            self.r: r,
            self.s_: s_,
            self.done: done,
            self.episode: episode
        })
        if episode % self.assign_interval==0:
            self.sess.run(self.assign_q_target)
        self.recorder.writer.add_summary(summaries, self.sess.run(self.global_step))