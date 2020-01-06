import numpy as np
import tensorflow as tf
import Nn
from utils.sth import sth
from Algorithms.tf2algos.base.off_policy import Off_Policy
from utils.expl_expt import ExplorationExploitationClass
from utils.tf2_utils import huber_loss


class IQN(Off_Policy):
    '''
    Implicit Quantile Networks, https://arxiv.org/abs/1806.06923
    Double DQN
    '''

    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,

                 online_quantiles=8,
                 target_quantiles=8,
                 select_quantiles=32,
                 quantiles_idx=64,
                 huber_delta=1.,
                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_episode=100,
                 assign_interval=2,
                 hidden_units={
                     'q_net': [128, 64],
                     'quantile': [128, 64],
                     'tile': [64]
                 },
                 **kwargs):
        assert not is_continuous, 'iqn only support discrete action space'
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim_or_list=a_dim_or_list,
            is_continuous=is_continuous,
            **kwargs)
        self.pi = tf.constant(np.pi)
        self.online_quantiles = online_quantiles
        self.target_quantiles = target_quantiles
        self.select_quantiles = select_quantiles
        self.quantiles_idx = quantiles_idx
        self.huber_delta = huber_delta
        self.assign_interval = assign_interval
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_episode=init2mid_annealing_episode,
                                                          max_episode=self.max_episode)
        self.visual_net = Nn.VisualNet('visual_net', self.visual_dim)
        self.q_net = Nn.iqn_net(self.s_dim, self.a_counts, self.quantiles_idx, 'q_net', hidden_units, visual_net=self.visual_net)
        self.q_target_net = Nn.iqn_net(self.s_dim, self.a_counts, self.quantiles_idx, 'q_target_net', hidden_units, visual_net=self.visual_net)
        self.update_target_net_weights(self.q_target_net.weights, self.q_net.weights)
        self.lr = tf.keras.optimizers.schedules.PolynomialDecay(lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr(self.episode))

    def show_logo(self):
        self.recorder.logger.info('''
　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　　　ｘｘｘ　　
　　　　ｘｘｘｘｘｘｘｘ　　　　　　ｘｘｘｘｘｘｘｘｘ　　　　　　ｘｘｘｘ　　　ｘｘｘ　　
　　　　　　ｘｘｘ　　　　　　　　　ｘｘｘｘ　　ｘｘｘｘ　　　　　ｘｘｘｘｘ　　ｘｘｘ　　
　　　　　　ｘｘｘ　　　　　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　ｘｘｘｘｘ　　ｘｘｘ　　
　　　　　　ｘｘｘ　　　　　　　　ｘｘｘｘ　　　　ｘｘｘ　　　　　ｘｘｘｘｘｘ　ｘｘｘ　　
　　　　　　ｘｘｘ　　　　　　　　ｘｘｘｘ　　　　ｘｘｘ　　　　　ｘｘｘｘｘｘｘｘｘｘ　　
　　　　　　ｘｘｘ　　　　　　　　ｘｘｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　ｘｘｘｘｘｘ　　
　　　　　　ｘｘｘ　　　　　　　　　ｘｘｘｘ　　ｘｘｘｘ　　　　　ｘｘｘ　ｘｘｘｘｘｘ　　
　　　　ｘｘｘｘｘｘｘｘ　　　　　　ｘｘｘｘｘｘｘｘｘ　　　　　　ｘｘｘ　　ｘｘｘｘｘ　　
　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　　ｘｘｘｘ　　
　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘ　　　　　　　　　　　　　　　　　　　
　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘ　　　　　　　　　　　　　　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　
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
            _, select_quantiles_tiled = self._generate_quantiles(   # [N*B, 64]
                batch_size=s.shape[0],
                quantiles_num=self.select_quantiles,
                quantiles_idx=self.quantiles_idx
            )
            _, q_values = self.q_net(s, visual_s, select_quantiles_tiled, quantiles_num=self.select_quantiles)  # [B, A]
        return tf.argmax(q_values, axis=-1)  # [B,]

    @tf.function
    def _generate_quantiles(self, batch_size, quantiles_num, quantiles_idx):
        with tf.device(self.device):
            _quantiles = tf.random.uniform([batch_size * quantiles_num, 1], minval=0, maxval=1)  # [N*B, 1]
            _quantiles_tiled = tf.tile(_quantiles, [1, quantiles_idx])  # [N*B, 1] => [N*B, 64]
            _quantiles_tiled = tf.cast(tf.range(quantiles_idx), tf.float32) * self.pi * _quantiles_tiled  # pi * i * tau [N*B, 64] * [64, ] => [N*B, 64]
            _quantiles_tiled = tf.cos(_quantiles_tiled)   # [N*B, 64]
            _quantiles = tf.reshape(_quantiles, [batch_size, quantiles_num, 1])    # [N*B, 1] => [B, N, 1]
            return _quantiles, _quantiles_tiled

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
                    self.update_target_net_weights(self.q_target_net.weights, self.q_net.weights)

                summaries.update(dict([
                    ['LEARNING_RATE/lr', self.lr(self.episode)]
                ]))
                self.write_training_summaries(self.global_step, summaries)

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, r, s_, visual_s_, done):
        s, visual_s, a, r, s_, visual_s_, done = self.cast(s, visual_s, a, r, s_, visual_s_, done)
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                quantiles, quantiles_tiled = self._generate_quantiles(   # [B, N, 1], [N*B, 64]
                    batch_size=s.shape[0],
                    quantiles_num=self.online_quantiles,
                    quantiles_idx=self.quantiles_idx
                )
                quantiles_value, q = self.q_net(s, visual_s, quantiles_tiled, quantiles_num=self.online_quantiles)    # [N, B, A], [B, A]
                _a = tf.reshape(tf.tile(a, [self.online_quantiles, 1]), [self.online_quantiles, -1, self.a_counts])  # [B, A] => [N*B, A] => [N, B, A]
                quantiles_value = tf.reduce_sum(quantiles_value * _a, axis=-1, keepdims=True)   # [N, B, A] => [N, B, 1]
                q_eval = tf.reduce_sum(q * a, axis=-1, keepdims=True)  # [B, A] => [B, 1]

                next_max_action = self._get_action(s_, visual_s_)   # [B,]
                next_max_action = tf.one_hot(tf.squeeze(next_max_action), self.a_counts, 1., 0., dtype=tf.float32)  # [B, A]
                _next_max_action = tf.reshape(tf.tile(next_max_action, [self.target_quantiles, 1]), [self.target_quantiles, -1, self.a_counts])  # [B, A] => [N'*B, A] => [N', B, A]
                _, target_quantiles_tiled = self._generate_quantiles(   # [N'*B, 64]
                    batch_size=s_.shape[0],
                    quantiles_num=self.target_quantiles,
                    quantiles_idx=self.quantiles_idx
                )

                target_quantiles_value, target_q = self.q_target_net(s_, visual_s_, target_quantiles_tiled, quantiles_num=self.target_quantiles)  # [N', B, A], [B, A]
                target_quantiles_value = tf.reduce_sum(target_quantiles_value * _next_max_action, axis=-1, keepdims=True)   # [N', B, A] => [N', B, 1]
                target_q = tf.reduce_sum(target_q * a, axis=-1, keepdims=True)  # [B, A] => [B, 1]
                q_target = tf.stop_gradient(r + self.gamma * (1 - done) * target_q)   # [B, 1]
                td_error = q_eval - q_target    # [B, 1]

                _r = tf.reshape(tf.tile(r, [self.target_quantiles, 1]), [self.target_quantiles, -1, 1])  # [B, 1] => [N'*B, 1] => [N', B, 1]
                _done = tf.reshape(tf.tile(done, [self.target_quantiles, 1]), [self.target_quantiles, -1, 1])    # [B, 1] => [N'*B, 1] => [N', B, 1]

                quantiles_value_target = tf.stop_gradient(_r + self.gamma * (1 - _done) * target_quantiles_value)   # [N', B, 1]
                quantiles_value_target = tf.transpose(quantiles_value_target, [1, 2, 0])    # [B, 1, N']
                quantiles_value_online = tf.transpose(quantiles_value, [1, 0, 2])   # [B, N, 1]
                quantile_error = quantiles_value_online - quantiles_value_target    # [B, N, 1] - [B, 1, N'] => [B, N, N']
                huber = huber_loss(quantile_error, delta=self.huber_delta)  # [B, N, N']
                huber_abs = tf.abs(quantiles - tf.where(quantile_error < 0, tf.ones_like(quantile_error), tf.zeros_like(quantile_error)))   # [B, N, 1] - [B, N, N'] => [B, N, N']
                loss = tf.reduce_mean(huber_abs * huber, axis=-1)  # [B, N, N'] => [B, N]
                loss = tf.reduce_sum(loss, axis=-1)  # [B, N] => [B, ]
                loss = tf.reduce_mean(loss * self.IS_w)  # [B, ] => 1
            grads = tape.gradient(loss, self.q_net.tv)
            self.optimizer.apply_gradients(
                zip(grads, self.q_net.tv)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/loss', loss],
                ['Statistics/q_max', tf.reduce_max(q_eval)],
                ['Statistics/q_min', tf.reduce_min(q_eval)],
                ['Statistics/q_mean', tf.reduce_mean(q_eval)]
            ])
