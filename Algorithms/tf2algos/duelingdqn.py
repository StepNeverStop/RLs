import numpy as np
import tensorflow as tf
import Nn
from utils.sth import sth
from .policy import Policy
from utils.expl_expt import ExplorationExploitationClass


class DDDQN(Policy):
    '''
    Dueling Double DQN
    '''

    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 action_type,
                 gamma=0.99,
                 max_episode=50000,
                 batch_size=128,
                 buffer_size=10000,
                 use_priority=False,
                 n_step=False,
                 base_dir=None,

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
                 logger2file=False,
                 out_graph=False):
        assert action_type == 'discrete', 'dueling double dqn only support discrete action space'
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim_or_list=a_dim_or_list,
            action_type=action_type,
            gamma=gamma,
            max_episode=max_episode,
            base_dir=base_dir,
            policy_mode='OFF',
            batch_size=batch_size,
            buffer_size=buffer_size,
            use_priority=use_priority,
            n_step=n_step)
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init, 
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_episode=init2mid_annealing_episode,
                                                          max_episode=max_episode)
        self.assign_interval = assign_interval
        self.dueling_net = Nn.critic_dueling(self.s_dim, self.visual_dim, self.a_counts, 'dueling_net', hidden_units)
        self.dueling_target_net = Nn.critic_dueling(self.s_dim, self.visual_dim, self.a_counts, 'dueling_target_net', hidden_units)
        self.update_target_net_weights(self.dueling_target_net.weights, self.dueling_net.weights)
        self.lr = tf.keras.optimizers.schedules.PolynomialDecay(lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr(self.episode))
        self.generate_recorder(
            logger2file=logger2file,
            model=self
        )
        self.recorder.logger.info('''
　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　ｘｘｘｘ　　　ｘｘｘｘ　　
　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘ　ｘｘｘｘ　　　　　　　ｘｘｘ　　　　ｘ　　　
　　　　ｘｘ　　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　ｘｘｘｘ　　　　　　ｘｘｘｘ　　　ｘ　　　
　　　　ｘｘ　　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘｘｘｘｘ　　ｘ　　　
　　　　ｘｘ　　　　　ｘｘ　　　　　　ｘｘ　　　　　ｘｘ　　　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘｘ　　　　　　ｘ　ｘｘｘｘ　ｘ　　　
　　　　ｘｘ　　　　　ｘｘ　　　　　　ｘｘ　　　　　ｘｘ　　　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘ　　ｘｘｘｘｘ　　　
　　　　ｘｘ　　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘ　　　ｘｘｘｘ　　　
　　　　ｘｘ　　　ｘｘｘｘ　　　　　　ｘｘ　　　ｘｘｘｘ　　　　　　ｘｘ　　　ｘｘｘｘ　　　　　ｘｘｘ　　　ｘｘｘ　　　　　　　ｘ　　　　ｘｘｘ　　　
　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　ｘｘｘ　　　　ｘｘ　　　
　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　　　　　　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘ　　　　　　　　　　　　　　　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘ　　　　　　　　　　　　　
        ''')

    def choose_action(self, s, visual_s):
        if np.random.uniform() < self.expl_expt_mng.get_esp(self.episode):
            a = np.random.randint(0, self.a_counts, len(s))
        else:
            a = self._get_action(s, visual_s).numpy()
        return sth.int2action_index(a, self.a_dim_or_list)

    def choose_inference_action(self, s, visual_s):
        if np.random.uniform() < self.expl_expt_mng.get_esp(self.episode, evaluation=True):
            a = np.random.randint(0, self.a_counts, len(s))
        else:
            a = self._get_action(s, visual_s).numpy()
        return sth.int2action_index(a, self.a_dim_or_list)

    @tf.function
    def _get_action(self, vector_input, visual_input):
        with tf.device(self.device):
            _, advs = self.dueling_net(vector_input, visual_input)
        return tf.argmax(advs, axis=1)

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        self.off_store(s, visual_s, a, r, s_, visual_s_, done)

    def learn(self, **kwargs):
        self.episode = kwargs['episode']
        self.recorder.writer.set_as_default()
        for i in range(kwargs['step']):
            if self.data.is_lg_batch_size:
                s, visual_s, a, r, s_, visual_s_, done = self.data.sample()
                if self.use_priority:
                    self.IS_w = self.data.get_IS_w()
                q_loss, td_error = self.train(s, visual_s, a, r, s_, visual_s_, done)
                if self.use_priority:
                    self.data.update(td_error, self.episode)
                if self.global_step % self.assign_interval == 0:
                    self.update_target_net_weights(self.dueling_target_net.weights, self.dueling_net.weights)
                tf.summary.experimental.set_step(self.global_step)
                tf.summary.scalar('LOSS/loss', q_loss)
                tf.summary.scalar('LEARNING_RATE/lr', self.lr(self.episode))
                self.recorder.writer.flush()

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, r, s_, visual_s_, done):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                v, adv = self.dueling_net(s, visual_s)
                average_adv = tf.reduce_mean(adv, axis=1, keepdims=True)
                v_next, adv_next = self.dueling_net(s_, visual_s_)
                next_max_action = tf.argmax(adv_next, axis=1, name='next_action_int')
                next_max_action_one_hot = tf.one_hot(tf.squeeze(next_max_action), self.a_counts, 1., 0., dtype=tf.float32)
                next_max_action_one_hot = tf.cast(next_max_action_one_hot, tf.float32)
                v_next_target, adv_next_target = self.dueling_target_net(s_, visual_s_)
                average_a_target_next = tf.reduce_mean(adv_next_target, axis=1, keepdims=True)
                q_eval = tf.reduce_sum(tf.multiply(v + adv - average_adv, a), axis=1, keepdims=True)
                q_target_next_max = tf.reduce_sum(
                    tf.multiply(v_next_target + adv_next_target - average_a_target_next, next_max_action_one_hot),
                    axis=1, keepdims=True)
                q_target = tf.stop_gradient(r + self.gamma * (1 - done) * q_target_next_max)
                td_error = q_eval - q_target
                q_loss = tf.reduce_mean(tf.square(td_error) * self.IS_w)
            grads = tape.gradient(q_loss, self.dueling_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.dueling_net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return q_loss, td_error
