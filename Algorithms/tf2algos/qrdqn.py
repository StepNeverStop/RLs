import Nn
import numpy as np
import tensorflow as tf
from utils.sth import sth
from Algorithms.tf2algos.base.off_policy import Off_Policy
from utils.expl_expt import ExplorationExploitationClass
from utils.tf2_utils import huber_loss


class QRDQN(Off_Policy):
    '''
    Quantile Regression DQN
    Distributional Reinforcement Learning with Quantile Regression, https://arxiv.org/abs/1710.10044
    No double, no dueling, no noisy net.
    '''

    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,

                 nums=20,
                 huber_delta=1.,
                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_episode=100,
                 assign_interval=1000,
                 hidden_units=[128, 128],
                 **kwargs):
        assert not is_continuous, 'qrdqn only support discrete action space'
        assert nums > 0
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim_or_list=a_dim_or_list,
            is_continuous=is_continuous,
            **kwargs)
        self.nums = nums
        self.huber_delta = huber_delta
        self.quantiles = tf.reshape(tf.constant((2 * np.arange(self.nums) + 1) / (2.0 * self.nums), dtype=tf.float32), [-1, self.nums]) # [1, N]
        self.batch_quantiles = tf.tile(self.quantiles, [self.a_counts, 1])  # [1, N] => [A, N]
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_episode=init2mid_annealing_episode,
                                                          max_episode=self.max_episode)
        self.assign_interval = assign_interval
        self.visual_net = Nn.VisualNet('visual_net', self.visual_dim)
        self.q_dist_net = Nn.qrdqn_distributional(self.s_dim, self.a_counts, self.nums, 'q_dist_net', hidden_units, visual_net=self.visual_net)
        self.q_target_dist_net = Nn.qrdqn_distributional(self.s_dim, self.a_counts, self.nums, 'q_target_dist_net', hidden_units, visual_net=self.visual_net)
        self.update_target_net_weights(self.q_target_dist_net.weights, self.q_dist_net.weights)
        self.lr = tf.keras.optimizers.schedules.PolynomialDecay(lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr(self.episode))
    
    def show_logo(self):
        self.recorder.logger.info('''
　　　　　ｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　ｘｘｘｘ　　　ｘｘｘｘ　　
　　　　ｘｘｘ　ｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘ　ｘｘｘｘ　　　　　　　ｘｘｘ　　　　ｘ　　　
　　　ｘｘｘ　　　ｘｘｘｘ　　　　　　ｘｘ　　ｘｘｘ　　　　　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　ｘｘｘｘ　　　　　　ｘｘｘｘ　　　ｘ　　　
　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘｘ　　ｘｘｘ　　　　　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘｘｘｘｘ　　ｘ　　　
　　　ｘｘ　　　　　ｘｘｘ　　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘｘ　　　　　　ｘ　ｘｘｘｘ　ｘ　　　
　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘ　　ｘｘｘｘｘ　　　
　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘｘ　ｘｘｘｘ　　　　　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘ　　　ｘｘｘｘ　　　
　　　ｘｘｘ　　　ｘｘｘ　　　　　　　ｘｘ　　ｘｘｘ　　　　　　　　ｘｘ　　　ｘｘｘｘ　　　　　ｘｘｘ　　　ｘｘｘ　　　　　　　ｘ　　　　ｘｘｘ　　　
　　　　ｘｘｘｘｘｘｘｘ　　　　　　ｘｘｘｘｘ　ｘｘｘｘ　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　ｘｘｘ　　　　ｘｘ　　　
　　　　　ｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘ　ｘｘｘｘ　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　　　　　　　　　　
　　　　　　　ｘｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘ　　　　　　　　　　　　　　　　　　　
　　　　　　　　　ｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘ　　　　　
        ''')

    def choose_action(self, s, visual_s, evaluation=False):
        if np.random.uniform() < self.expl_expt_mng.get_esp(self.episode, evaluation=evaluation):
            a = np.random.randint(0, self.a_counts, len(s))
        else:
            a = self._get_action(s, visual_s).numpy()
        return sth.int2action_index(a, self.a_dim_or_list)

    @tf.function
    def _get_action(self, s, visual_s):
        s, visual_s = self.cast(s, visual_s)
        with tf.device(self.device):
            q = self.get_q(s, visual_s)  # [B, A]
        return tf.argmax(q, axis=-1)  # [B, 1]

    def learn(self, **kwargs):
        self.episode = kwargs['episode']
        for i in range(kwargs['step']):
            if self.data.is_lg_batch_size:
                s, visual_s, a, r, s_, visual_s_, done = self.data.sample()
                if self.use_priority:
                    self.IS_w = self.data.get_IS_w()
                td_error, summaries = self.train(s, visual_s, a, r, s_, visual_s_, done)
                if self.use_priority:
                    td_error = np.squeeze(td_error.numpy())
                    self.data.update(td_error, self.episode)
                if self.global_step % self.assign_interval == 0:
                    self.update_target_net_weights(self.q_target_dist_net.weights, self.q_dist_net.weights)
                summaries.update(dict([['LEARNING_RATE/lr', self.lr(self.episode)]]))
                self.write_training_summaries(self.global_step, summaries)

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, r, s_, visual_s_, done):
        s, visual_s, a, r, s_, visual_s_, done = self.cast(s, visual_s, a, r, s_, visual_s_, done)
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                indexs = tf.reshape(tf.range(s.shape[0]), [-1, 1])  # [B, 1]
                q_dist = self.q_dist_net(s, visual_s)  # [B, A, N]
                q_dist = tf.transpose(tf.reduce_sum(tf.transpose(q_dist, [2, 0, 1]) * a, axis=-1), [1, 0])  # [B, N]
                target_q_dist = self.q_target_dist_net(s_, visual_s_)  # [B, A, N]
                target_q = tf.reduce_sum(self.batch_quantiles * target_q_dist, axis=-1)  # [B, A, N] => [B, A]
                a_ = tf.reshape(tf.cast(tf.argmax(target_q, axis=-1), dtype=tf.int32), [-1, 1])  # [B, 1]
                target_q_dist = tf.gather_nd(target_q_dist, tf.concat([indexs, a_], axis=-1))   # [B, N]
                target = tf.tile(r, tf.constant([1, self.nums])) \
                    + self.gamma * tf.multiply(self.quantiles,   # [1, N]
                                               (1.0 - tf.tile(done, tf.constant([1, self.nums]))))  # [B, N], [1, N]* [B, N] = [B, N]
                q_eval = tf.reduce_sum(q_dist * self.quantiles, axis=-1)    # [B, 1]
                q_target = tf.reduce_sum(target * self.quantiles, axis=-1)  # [B, 1]
                td_error = q_eval - q_target    # [B, 1]

                quantile_error = tf.expand_dims(q_dist, axis=-1) - tf.expand_dims(target, axis=1) # [B, N, 1] - [B, 1, N] => [B, N, N]
                huber = huber_loss(quantile_error, delta=self.huber_delta)  # [B, N, N]
                huber_abs = tf.abs(self.quantiles - tf.where(quantile_error < 0, tf.ones_like(quantile_error), tf.zeros_like(quantile_error)))   # [1, N] - [B, N, N] => [B, N, N]
                loss = tf.reduce_mean(huber_abs * huber, axis=-1)  # [B, N, N] => [B, N]
                loss = tf.reduce_sum(loss, axis=-1)  # [B, N] => [B, ]
                loss = tf.reduce_mean(loss * self.IS_w)  # [B, ] => 1
            grads = tape.gradient(loss, self.q_dist_net.tv)
            self.optimizer.apply_gradients(
                zip(grads, self.q_dist_net.tv)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/loss', loss],
                ['Statistics/q_max', tf.reduce_max(q_eval)],
                ['Statistics/q_min', tf.reduce_min(q_eval)],
                ['Statistics/q_mean', tf.reduce_mean(q_eval)]
            ])

    @tf.function(experimental_relax_shapes=True)
    def get_q(self, s, visual_s):
        with tf.device(self.device):
            return tf.reduce_sum(self.batch_quantiles * self.q_dist_net(s, visual_s), axis=-1)  # [B, A, N] => [B, A]
