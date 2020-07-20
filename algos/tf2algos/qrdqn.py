import rls
import numpy as np
import tensorflow as tf
from algos.tf2algos.base.off_policy import make_off_policy_class
from utils.expl_expt import ExplorationExploitationClass
from utils.tf2_utils import huber_loss


class QRDQN(make_off_policy_class(mode='share')):
    '''
    Quantile Regression DQN
    Distributional Reinforcement Learning with Quantile Regression, https://arxiv.org/abs/1710.10044
    No double, no dueling, no noisy net.
    '''

    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim,
                 is_continuous,

                 nums=20,
                 huber_delta=1.,
                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=1000,
                 hidden_units=[128, 128],
                 **kwargs):
        assert not is_continuous, 'qrdqn only support discrete action space'
        assert nums > 0
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim=a_dim,
            is_continuous=is_continuous,
            **kwargs)
        self.nums = nums
        self.huber_delta = huber_delta
        self.quantiles = tf.reshape(tf.constant((2 * np.arange(self.nums) + 1) / (2.0 * self.nums), dtype=tf.float32), [-1, self.nums])  # [1, N]
        self.batch_quantiles = tf.tile(self.quantiles, [self.a_dim, 1])  # [1, N] => [A, N]
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval

        def _net(): return rls.qrdqn_distributional(self.feat_dim, self.a_dim, self.nums, hidden_units)
        self.q_dist_net = _net()
        self.q_target_dist_net = _net()
        self.critic_tv = self.q_dist_net.trainable_variables + self.other_tv
        self.update_target_net_weights(self.q_target_dist_net.weights, self.q_dist_net.weights)
        self.lr = self.init_lr(lr)
        self.optimizer = self.init_optimizer(self.lr)
        self.model_recorder(dict(
            model=self.q_dist_net,
            optimizer=self.optimizer
        ))

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
        if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
            a = np.random.randint(0, self.a_dim, self.n_agents)
        else:
            a, self.cell_state = self._get_action(s, visual_s, self.cell_state)
            a = a.numpy()
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state):
        with tf.device(self.device):
            feat, cell_state = self.get_feature(s, visual_s, cell_state=cell_state, record_cs=True)
            q = self.get_q(feat)  # [B, A]
        return tf.argmax(q, axis=-1), cell_state  # [B, 1]

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        def _update():
            if self.global_step % self.assign_interval == 0:
                self.update_target_net_weights(self.q_target_dist_net.weights, self.q_dist_net.weights)
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'train_function': self.train,
                'update_function': _update,
                'summary_dict': dict([['LEARNING_RATE/lr', self.lr(self.train_step)]])
            })

    @tf.function(experimental_relax_shapes=True)
    def train(self, memories, isw, crsty_loss, cell_state):
        ss, vvss, a, r, done = memories
        batch_size = tf.shape(a)[0]
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                indexs = tf.reshape(tf.range(batch_size), [-1, 1])  # [B, 1]
                q_dist = self.q_dist_net(feat)  # [B, A, N]
                q_dist = tf.transpose(tf.reduce_sum(tf.transpose(q_dist, [2, 0, 1]) * a, axis=-1), [1, 0])  # [B, N]
                target_q_dist = self.q_target_dist_net(feat_)  # [B, A, N]
                target_q = tf.reduce_sum(self.batch_quantiles * target_q_dist, axis=-1)  # [B, A, N] => [B, A]
                a_ = tf.reshape(tf.cast(tf.argmax(target_q, axis=-1), dtype=tf.int32), [-1, 1])  # [B, 1]
                target_q_dist = tf.gather_nd(target_q_dist, tf.concat([indexs, a_], axis=-1))   # [B, N]
                target = tf.tile(r, tf.constant([1, self.nums])) \
                    + self.gamma * tf.multiply(self.quantiles,   # [1, N]
                                               (1.0 - tf.tile(done, tf.constant([1, self.nums]))))  # [B, N], [1, N]* [B, N] = [B, N]
                q_eval = tf.reduce_sum(q_dist * self.quantiles, axis=-1)    # [B, 1]
                q_target = tf.reduce_sum(target * self.quantiles, axis=-1)  # [B, 1]
                td_error = q_eval - q_target    # [B, 1]

                quantile_error = tf.expand_dims(q_dist, axis=-1) - tf.expand_dims(target, axis=1)  # [B, N, 1] - [B, 1, N] => [B, N, N]
                huber = huber_loss(quantile_error, delta=self.huber_delta)  # [B, N, N]
                huber_abs = tf.abs(self.quantiles - tf.where(quantile_error < 0, tf.ones_like(quantile_error), tf.zeros_like(quantile_error)))   # [1, N] - [B, N, N] => [B, N, N]
                loss = tf.reduce_mean(huber_abs * huber, axis=-1)  # [B, N, N] => [B, N]
                loss = tf.reduce_sum(loss, axis=-1)  # [B, N] => [B, ]
                loss = tf.reduce_mean(loss * isw) + crsty_loss  # [B, ] => 1
            grads = tape.gradient(loss, self.critic_tv)
            self.optimizer.apply_gradients(
                zip(grads, self.critic_tv)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/loss', loss],
                ['Statistics/q_max', tf.reduce_max(q_eval)],
                ['Statistics/q_min', tf.reduce_min(q_eval)],
                ['Statistics/q_mean', tf.reduce_mean(q_eval)]
            ])

    @tf.function(experimental_relax_shapes=True)
    def get_q(self, feat):
        with tf.device(self.device):
            return tf.reduce_sum(self.batch_quantiles * self.q_dist_net(feat), axis=-1)  # [B, A, N] => [B, A]
