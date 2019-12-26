import Nn
import numpy as np
import tensorflow as tf
from utils.sth import sth
from Algorithms.tf2algos.base.off_policy import Off_Policy
from utils.expl_expt import ExplorationExploitationClass


class RAINBOW(Off_Policy):
    '''
    Dueling Double DQN
    '''

    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,

                 v_min=-10,
                 v_max=10,
                 atoms=51,
                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_episode=100,
                 assign_interval=2,
                 hidden_units={
                     'share': [128],
                     'v': [128],
                     'adv': [128]
                 },
                 **kwargs):
        assert not is_continuous, 'rainbow only support discrete action space'
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim_or_list=a_dim_or_list,
            is_continuous=is_continuous,
            **kwargs)
        self.v_min = v_min
        self.v_max = v_max
        self.atoms = atoms
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self.z = np.asarray([self.v_min + i * self.delta_z for i in range(self.atoms)])
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_episode=init2mid_annealing_episode,
                                                          max_episode=self.max_episode)
        self.assign_interval = assign_interval
        self.rainbow_net = Nn.rainbow_dueling(self.s_dim, self.visual_dim, self.a_counts, self.atoms, 'rainbow_net', hidden_units)
        self.rainbow_target_net = Nn.rainbow_dueling(self.s_dim, self.visual_dim, self.a_counts, self.atoms, 'rainbow_target_net', hidden_units)
        self.update_target_net_weights(self.rainbow_target_net.weights, self.rainbow_net.weights)
        self.lr = tf.keras.optimizers.schedules.PolynomialDecay(lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr(self.episode))
        self.recorder.logger.info('''
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　
　　　ｘｘｘｘｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　
　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　
　　　　ｘｘ　　ｘｘｘ　　　　　　　　　　　ｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘ　ｘｘ　　　　　　　　　　　ｘｘｘｘ　　　　　　　　　　　　　　　　　　　　
　　　　ｘｘ　　ｘｘｘ　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　ｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘ　ｘｘ　　　
　　　　ｘｘｘｘｘｘ　　　　　　　　　　ｘｘ　ｘｘ　　　　　　　　　　　ｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘ　ｘｘｘ　　　　　　　ｘｘｘ　　ｘｘ　　　　　　　ｘｘｘ　ｘｘ　ｘｘ　　　
　　　　ｘｘｘｘｘｘ　　　　　　　　　　　ｘｘｘｘ　　　　　　　　　　　　ｘｘ　　　　　　　　　　　ｘｘ　ｘｘ　　　　　　　　　　ｘｘ　　　ｘｘ　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　ｘｘ　ｘｘ　ｘｘ　　　
　　　　ｘｘ　ｘｘｘｘ　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　　ｘｘ　　　　　　　　　　　ｘｘ　ｘｘ　　　　　　　　　　ｘｘ　　　ｘｘ　　　　　　　ｘｘｘ　　ｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　
　　　　ｘｘ　　ｘｘｘ　　　　　　　　　ｘｘ　ｘｘ　ｘ　　　　　　　　　　ｘｘ　　　　　　　　　　　ｘｘ　ｘｘ　　　　　　　　　　ｘｘ　　ｘｘｘ　　　　　　　　ｘｘ　ｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　
　　　ｘｘｘｘｘ　ｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　ｘｘｘ　　　　　　　　　　ｘｘｘ　ｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　ｘｘ　ｘｘ　　　　　
　　　ｘｘｘｘｘ　ｘｘｘｘ　　　　　　　ｘｘｘ　ｘｘ　　　　　　　　　　ｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　　ｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘ　ｘｘ　　　　　　　　
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
            _, advs = self.rainbow_net(s, visual_s)
        return tf.argmax(advs, axis=1)

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
                    self.update_target_net_weights(self.rainbow_target_net.weights, self.rainbow_net.weights)
                summaries.update(dict([
                    ['LEARNING_RATE/lr', self.lr(self.episode)]
                ]))
                self.write_training_summaries(self.global_step, summaries)

    # @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, r, s_, visual_s_, done):
        # batch_size = s.shape[0]
        # indexs = list(range(batch_size))
        # s, visual_s, a, r, s_, visual_s_, done = self.cast(s, visual_s, a, r, s_, visual_s_, done)
        # with tf.device(self.device):
        #     with tf.GradientTape() as tape:
        #         q_dist = self.q_dist_net(s, visual_s, a)
        #         list_q_ = [self.get_target_q(s_, visual_s_, tf.one_hot([i] * batch_size, self.a_counts)) for i in range(self.a_counts)]  # [a_counts, batch_size]
        #         a_ = tf.argmax(list_q_, axis=0)                                                             # [batch_size, ]
        #         m = np.zeros(shape=(batch_size, self.atoms))                                                # [batch_size, atoms]
        #         target_q_dist = self.q_target_dist_net(s_, visual_s_, tf.one_hot(a_, self.a_counts))               # [batch_size, atoms]
        #         for j in range(self.atoms):
        #             Tz = tf.squeeze(tf.clip_by_value(r + self.gamma * self.z[j], self.v_min, self.v_max))   # [batch_size, ]
        #             bj = (Tz - self.v_min) / self.delta_z           # [batch_size, ]
        #             l, u = tf.math.floor(bj), tf.math.ceil(bj)      # [batch_size, ]
        #             pj = target_q_dist[:, j]                               # [batch_size, ]
        #             m[indexs, tf.cast(l, tf.int32)] += pj * (u - bj)
        #             m[indexs, tf.cast(u, tf.int32)] += pj * (bj - l)
        #         cross_entropy = -tf.reduce_sum(m * tf.math.log(q_dist), axis=1)  # [batch_size, 1]
        #         loss = tf.reduce_mean(cross_entropy) * self.IS_w

        s, visual_s, a, r, s_, visual_s_, done = self.cast(s, visual_s, a, r, s_, visual_s_, done)
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                v, adv = self.rainbow_net(s, visual_s)
                average_adv = tf.reduce_mean(adv, axis=1, keepdims=True)
                v_next, adv_next = self.rainbow_net(s_, visual_s_)
                next_max_action = tf.argmax(adv_next, axis=1, name='next_action_int')
                next_max_action_one_hot = tf.one_hot(tf.squeeze(next_max_action), self.a_counts, 1., 0., dtype=tf.float32)
                next_max_action_one_hot = tf.cast(next_max_action_one_hot, tf.float32)
                v_next_target, adv_next_target = self.rainbow_target_net(s_, visual_s_)
                average_a_target_next = tf.reduce_mean(adv_next_target, axis=1, keepdims=True)
                q_eval = tf.reduce_sum(tf.multiply(v + adv - average_adv, a), axis=1, keepdims=True)
                q_target_next_max = tf.reduce_sum(
                    tf.multiply(v_next_target + adv_next_target - average_a_target_next, next_max_action_one_hot),
                    axis=1, keepdims=True)
                q_target = tf.stop_gradient(r + self.gamma * (1 - done) * q_target_next_max)
                td_error = q_eval - q_target
                q_loss = tf.reduce_mean(tf.square(td_error) * self.IS_w)
            grads = tape.gradient(q_loss, self.rainbow_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.rainbow_net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/loss', q_loss],
                ['Statistics/v_mean', tf.reduce_max(v)],
                ['Statistics/advantage_mean', tf.reduce_max(adv)],
                ['Statistics/q_max', tf.reduce_max(q_eval)],
                ['Statistics/q_min', tf.reduce_min(q_eval)],
                ['Statistics/q_mean', tf.reduce_mean(q_eval)]
            ])
