import Nn
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from Algorithms.tf2algos.base.off_policy import Off_Policy
from utils.expl_expt import ExplorationExploitationClass


class OC(Off_Policy):
    '''
    The Option-Critic Architecture. http://arxiv.org/abs/1609.05140
    '''
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,

                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_episode=100,
                 options_num=4,
                 ent_coff=0.01,
                 double_q=False,
                 use_baseline=True,
                 terminal_mask=True,
                 termination_regularizer=0.01,
                 assign_interval=1000,
                 hidden_units=[32, 32],
                 **kwargs):
        assert not is_continuous, 'option-critic only support discrete action space'
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim_or_list=a_dim_or_list,
            is_continuous=is_continuous,
            **kwargs)
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_episode=init2mid_annealing_episode,
                                                          max_episode=self.max_episode)
        self.assign_interval = assign_interval
        self.options_num = options_num
        self.termination_regularizer = termination_regularizer
        self.ent_coff = ent_coff
        self.use_baseline = use_baseline
        self.terminal_mask = terminal_mask
        self.double_q = double_q

        _q_net = lambda : Nn.oc(self.rnn_net.hdim, self.a_counts, self.options_num, hidden_units)

        self.q_net = _q_net()
        self.q_target_net = _q_net()
        self.critic_tv = self.q_net.trainable_variables + self.other_tv
        self.update_target_net_weights(self.q_target_net.weights, self.q_net.weights)
        self.lr = tf.keras.optimizers.schedules.PolynomialDecay(lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr(self.episode), clipvalue=5.)

        self.model_recorder(dict(
            model=self.q_net,
            optimizer=self.optimizer
        ))

    def show_logo(self):
        self.recorder.logger.info('''
　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘｘｘ　　　
　　　　ｘｘｘ　ｘｘｘｘ　　　　　　　ｘｘｘｘ　ｘｘｘ　　　
　　　ｘｘｘ　　　ｘｘｘ　　　　　　ｘｘｘｘ　　　　ｘ　　　
　　　ｘｘ　　　　　ｘｘｘ　　　　　ｘｘｘ　　　　　ｘ　　　
　　　ｘｘ　　　　　ｘｘｘ　　　　　ｘｘｘ　　　　　　　　　
　　　ｘｘ　　　　　ｘｘｘ　　　　　ｘｘｘ　　　　　　　　　
　　　ｘｘ　　　　　ｘｘｘ　　　　　ｘｘｘ　　　　　　　　　
　　　ｘｘｘ　　　ｘｘｘ　　　　　　　ｘｘｘ　　　　ｘ　　　
　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　
　　　　　ｘｘｘｘｘ　　　　　　　　　　　ｘｘｘｘｘ
        ''')

    def choose_action(self, s, visual_s, evaluation=False):
        if np.random.uniform() < self.expl_expt_mng.get_esp(self.episode, evaluation=evaluation):
            self.options = tf.constant(np.random.randint(0, self.options_num, len(s) or len(visual_s)), dtype=tf.int32)
        a, self.cell_state = self._get_action(s, visual_s, self.cell_state, self.options)
        a = a.numpy()
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state, options):
        with tf.device(self.device):
            feat, cell_state = self.get_feature(s, visual_s, cell_state=cell_state, record_cs=True, train=False)
            q, pi, beta = self.q_net(feat)  # [B, P], [B, P, A], [B, P]
            op = tf.expand_dims(tf.one_hot(options, self.options_num, dtype=tf.float32), axis=-1)  # [B, P, 1]
            pi = tf.reduce_sum(pi * op, axis=1) # [B, A]
            dist = tfp.distributions.Categorical(logits=pi) # [B, ]
            a = dist.sample()
        return a, cell_state

    def learn(self, **kwargs):
        self.episode = kwargs['episode']
        def _update():
            if self.global_step % self.assign_interval == 0:
                self.update_target_net_weights(self.q_target_net.weights, self.q_net.weights)
        for i in range(kwargs['step']):
            self._learn(function_dict={
                'train_function': self.train,
                'update_function': _update,
                'summary_dict': dict([['LEARNING_RATE/lr', self.lr(self.episode)]]),
                'sample_data_list': ['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done', 'options'],
                'train_data_list': ['ss', 'vvss', 'a', 'r', 'done', 'options']
            })

    @tf.function(experimental_relax_shapes=True)
    def train(self, memories, isw, crsty_loss, cell_state):
        ss, vvss, a, r, done, options = memories
        options = tf.cast(options, tf.int32)
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                q, pi, beta = self.q_net(feat)  # [B, P], [B, P, A], [B, P]
                q_next, pi_next, beta_next = self.q_target_net(feat_)   # [B, P], [B, P, A], [B, P]
                options_onehot = tf.one_hot(options, self.options_num, dtype=tf.float32)    # [B,] => [B, P]

                q_s = qu_eval = tf.reduce_sum(q * options_onehot, axis=-1, keepdims=True) # [B, 1]
                beta_s_ = tf.reduce_sum(beta_next * options_onehot, axis=-1, keepdims=True) # [B, 1]
                q_s_ = tf.reduce_sum(q_next * options_onehot, axis=-1, keepdims=True)   # [B, 1]
                # https://github.com/jeanharb/option_critic/blob/5d6c81a650a8f452bc8ad3250f1f211d317fde8c/neural_net.py#L94
                if self.double_q:
                    q_, pi_, beta_ = self.q_net(feat)  # [B, P], [B, P, A], [B, P]
                    max_a_idx = tf.one_hot(tf.argmax(q_, axis=-1), self.options_num, dtype=tf.float32)  # [B, P] => [B, ] => [B, P]
                    q_s_max = tf.reduce_sum(q_next * max_a_idx, axis=-1, keepdims=True)   # [B, 1]
                else:
                    q_s_max = tf.reduce_max(q_next, axis=-1, keepdims=True)   # [B, 1]
                u_target = (1 - beta_s_) * q_s_ + beta_s_ * q_s_max   # [B, 1]
                qu_target = tf.stop_gradient(r + self.gamma * (1 - done) * u_target)
                td_error =  qu_target - qu_eval     # gradient : q
                q_loss = tf.square(td_error)        # [B, 1]

                # https://github.com/jeanharb/option_critic/blob/5d6c81a650a8f452bc8ad3250f1f211d317fde8c/neural_net.py#L130
                if self.use_baseline:
                    adv = tf.stop_gradient(qu_target - qu_eval)
                else:
                    adv = tf.stop_gradient(qu_target)
                options_onehot_expanded = tf.expand_dims(options_onehot, axis=-1)   # [B, P] => [B, P, 1]
                pi = tf.reduce_sum(pi * options_onehot_expanded, axis=1) # [B, P, A] => [B, A]
                log_pi = tf.nn.log_softmax(pi, axis=-1) # [B, A]
                entropy = -tf.reduce_sum(tf.exp(log_pi) * log_pi, axis=1, keepdims=True)    # [B, 1]
                log_p = tf.reduce_sum(a * log_pi, axis=-1, keepdims=True)   # [B, 1]
                pi_loss = -(log_p * adv + self.ent_coff * entropy)              # [B, 1] * [B, 1] => [B, 1]

                beta_s = tf.reduce_sum(beta * options_onehot, axis=-1, keepdims=True)   # [B, 1]
                v_s = tf.reduce_max(q, axis=-1, keepdims=True)   # [B, 1]
                beta_loss = beta_s * tf.stop_gradient(q_s - v_s + self.termination_regularizer)   # [B, 1]
                # https://github.com/lweitkamp/option-critic-pytorch/blob/0c57da7686f8903ed2d8dded3fae832ee9defd1a/option_critic.py#L238
                if self.terminal_mask:
                    beta_loss *= (1 - done)

                loss = tf.reduce_mean((q_loss + pi_loss + beta_loss) * isw) + crsty_loss

            grads = tape.gradient(loss, self.critic_tv)
            self.optimizer.apply_gradients(
                zip(grads, self.critic_tv)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/loss', loss],
                ['LOSS/q_loss', tf.reduce_mean(q_loss)],
                ['LOSS/pi_loss', tf.reduce_mean(pi_loss)],
                ['LOSS/beta_loss', tf.reduce_mean(beta_loss)],
                ['Statistics/q_max', tf.reduce_max(q_s)],
                ['Statistics/q_min', tf.reduce_min(q_s)],
                ['Statistics/q_mean', tf.reduce_mean(q_s)]
            ])

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        """
        for off-policy training, use this function to store <s, a, r, s_, done> into ReplayBuffer.
        """
        assert isinstance(a, np.ndarray), "store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store need done type is np.ndarray"
        self.data.add(
            s,
            visual_s,
            a,
            r[:, np.newaxis],   # 升维
            s_,
            visual_s_,
            done[:, np.newaxis], # 升维
            self.options
        )

    def no_op_store(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "no_op_store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "no_op_store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "no_op_store need done type is np.ndarray"
        self.data.add(
            s,
            visual_s,
            a,
            r[:, np.newaxis],
            s_,
            visual_s_,
            done[:, np.newaxis],
            self.options
        )