import rls
import numpy as np
import tensorflow as tf
from algos.tf2algos.base.off_policy import make_off_policy_class
from utils.expl_expt import ExplorationExploitationClass
from common.decorator import lazy_property


class C51(make_off_policy_class(mode='share')):
    '''
    Category 51, https://arxiv.org/abs/1707.06887
    No double, no dueling, no noisy net.
    '''

    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim,
                 is_continuous,

                 v_min=-10,
                 v_max=10,
                 atoms=51,
                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=1000,
                 hidden_units=[128, 128],
                 **kwargs):
        assert not is_continuous, 'c51 only support discrete action space'
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim=a_dim,
            is_continuous=is_continuous,
            **kwargs)
        self.v_min = v_min
        self.v_max = v_max
        self.atoms = atoms
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self.z = tf.reshape(tf.constant([self.v_min + i * self.delta_z for i in range(self.atoms)], dtype=tf.float32), [-1, self.atoms])  # [1, N]
        self.zb = tf.tile(self.z, tf.constant([self.a_dim, 1]))  # [A, N]
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval

        def _net(): return rls.c51_distributional(self.feat_dim, self.a_dim, self.atoms, hidden_units)

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
　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　ｘｘｘ　　　　　　
　　　　ｘｘｘｘ　ｘｘｘ　　　　　　　　　ｘｘｘｘ　　　　　　　　　　ｘｘｘｘ　　　　　　
　　　ｘｘｘｘ　　　　ｘ　　　　　　　　ｘｘｘｘ　　　　　　　　　　　　　ｘｘ　　　　　　
　　　ｘｘｘ　　　　　ｘ　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　　ｘｘ　　　　　　
　　　ｘｘｘ　　　　　　　　　　　　　　　　ｘｘｘ　　　　　　　　　　　　ｘｘ　　　　　　
　　　ｘｘｘ　　　　　　　　　　　　　　　　　ｘｘｘ　　　　　　　　　　　ｘｘ　　　　　　
　　　ｘｘｘ　　　　　　　　　　　　　　　　　　ｘｘ　　　　　　　　　　　ｘｘ　　　　　　
　　　　ｘｘｘ　　　　ｘ　　　　　　　　ｘｘ　ｘｘ　　　　　　　　　　　　ｘｘ　　　　　　
　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　ｘｘｘｘ　　　　　
　　　　　　ｘｘｘｘｘ　　　　　　　　　　ｘ　　　　　　　　　　　　　　ｘｘｘｘ　　　　
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
                q_eval = tf.reduce_sum(q_dist * self.z, axis=-1)
                target_q_dist = self.q_target_dist_net(feat_)  # [B, A, N]
                target_q = tf.reduce_sum(self.zb * target_q_dist, axis=-1)  # [B, A, N] => [B, A]
                a_ = tf.reshape(tf.cast(tf.argmax(target_q, axis=-1), dtype=tf.int32), [-1, 1])  # [B, 1]
                target_q_dist = tf.gather_nd(target_q_dist, tf.concat([indexs, a_], axis=-1))   # [B, N]
                target = tf.tile(r, tf.constant([1, self.atoms])) \
                    + self.gamma * tf.multiply(self.z,   # [1, N]
                                               (1.0 - tf.tile(done, tf.constant([1, self.atoms]))))  # [B, N], [1, N]* [B, N] = [B, N]
                target = tf.clip_by_value(target, self.v_min, self.v_max)  # [B, N]
                b = (target - self.v_min) / self.delta_z  # [B, N]
                u, l = tf.math.ceil(b), tf.math.floor(b)  # [B, N]
                u_id, l_id = tf.cast(u, tf.int32), tf.cast(l, tf.int32)  # [B, N]
                u_minus_b, b_minus_l = u - b, b - l  # [B, N]
                index_help = tf.tile(indexs, tf.constant([1, self.atoms]))  # [B, N]
                index_help = tf.expand_dims(index_help, -1)  # [B, N, 1]
                u_id = tf.concat([index_help, tf.expand_dims(u_id, -1)], axis=-1)    # [B, N, 2]
                l_id = tf.concat([index_help, tf.expand_dims(l_id, -1)], axis=-1)    # [B, N, 2]
                _cross_entropy = tf.stop_gradient(target_q_dist * u_minus_b) * tf.math.log(tf.gather_nd(q_dist, l_id)) \
                    + tf.stop_gradient(target_q_dist * b_minus_l) * tf.math.log(tf.gather_nd(q_dist, u_id))  # [B, N]
                # tf.debugging.check_numerics(_cross_entropy, '_cross_entropy')
                cross_entropy = -tf.reduce_sum(_cross_entropy, axis=-1)  # [B,]
                # tf.debugging.check_numerics(cross_entropy, 'cross_entropy')
                loss = tf.reduce_mean(cross_entropy * isw) + crsty_loss
                td_error = cross_entropy
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
            return tf.reduce_sum(self.zb * self.q_dist_net(feat), axis=-1)  # [B, A, N] => [B, A]
