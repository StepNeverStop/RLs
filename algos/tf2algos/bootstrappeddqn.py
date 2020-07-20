import rls
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from algos.tf2algos.base.off_policy import make_off_policy_class
from utils.expl_expt import ExplorationExploitationClass


class BootstrappedDQN(make_off_policy_class(mode='share')):
    '''
    Deep Exploration via Bootstrapped DQN, http://arxiv.org/abs/1602.04621
    '''

    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim,
                 is_continuous,

                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=1000,
                 head_num=4,
                 hidden_units=[32, 32],
                 **kwargs):
        assert not is_continuous, 'Bootstrapped DQN only support discrete action space'
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim=a_dim,
            is_continuous=is_continuous,
            **kwargs)
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval
        self.head_num = head_num
        self._probs = [1. / head_num for _ in range(head_num)]
        self.now_head = 0

        def _q_net(): return rls.critic_q_bootstrap(self.feat_dim, self.a_dim, self.head_num, hidden_units)

        self.q_net = _q_net()
        self.q_target_net = _q_net()
        self.critic_tv = self.q_net.trainable_variables + self.other_tv
        self.update_target_net_weights(self.q_target_net.weights, self.q_net.weights)
        self.lr = self.init_lr(lr)
        self.optimizer = self.init_optimizer(self.lr)

        self.model_recorder(dict(
            model=self.q_net,
            optimizer=self.optimizer
        ))

    def show_logo(self):
        self.recorder.logger.info('''
　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　ｘｘｘｘ　　　ｘｘｘｘ　　
　　　　　ｘｘ　ｘｘｘｘ　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘ　ｘｘｘｘ　　　　　　　ｘｘｘ　　　　ｘ　　　
　　　　　ｘｘ　　ｘｘｘ　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　ｘｘｘｘ　　　　　　ｘｘｘｘ　　　ｘ　　　
　　　　　ｘｘ　　ｘｘｘ　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘｘｘｘｘ　　ｘ　　　
　　　　　ｘｘｘｘｘｘ　　　　　　ｘｘｘ　ｘｘｘｘ　ｘｘｘ　　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘｘ　　　　　　ｘ　ｘｘｘｘ　ｘ　　　
　　　　　ｘｘ　ｘｘｘｘ　　　　　ｘｘｘ　ｘｘｘｘ　ｘｘｘ　　　　　ｘｘ　　　　　ｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘ　　ｘｘｘｘｘ　　　
　　　　　ｘｘ　　ｘｘｘ　　　　　ｘｘｘ　　ｘｘ　　ｘｘｘ　　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘｘ　　　　ｘｘｘ　　　　　　ｘ　　　ｘｘｘｘ　　　
　　　　　ｘｘ　　　ｘｘ　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　ｘｘｘｘ　　　　　ｘｘｘ　　　ｘｘｘ　　　　　　　ｘ　　　　ｘｘｘ　　　
　　　　　ｘｘ　ｘｘｘｘ　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　ｘｘｘ　　　　ｘｘ　　　
　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　　　　　　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘ　　　　　　　　　　　　　　　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘ　
        ''')

    def reset(self):
        super().reset()
        self.now_head = np.random.randint(self.head_num)

    def choose_action(self, s, visual_s, evaluation=False):
        if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
            a = np.random.randint(0, self.a_dim, self.n_agents)
        else:
            q, self.cell_state = self._get_action(s, visual_s, self.cell_state)
            q = q.numpy()
            a = np.argmax(q[self.now_head], axis=1)  # [H, B, A] => [B, A] => [B, ]
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state):
        with tf.device(self.device):
            feat, cell_state = self.get_feature(s, visual_s, cell_state=cell_state, record_cs=True)
            q_values = self.q_net(feat)  # [H, B, A]
        return q_values, cell_state

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')

        def _update():
            if self.global_step % self.assign_interval == 0:
                self.update_target_net_weights(self.q_target_net.weights, self.q_net.weights)
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
                q = self.q_net(feat)    # [H, B, A]
                q_next = self.q_target_net(feat_)   # [H, B, A]
                q_eval = tf.reduce_sum(tf.multiply(q, a), axis=-1, keepdims=True)    # [H, B, A] * [B, A] => [H, B, 1]
                q_target = tf.stop_gradient(r + self.gamma * (1 - done) * tf.reduce_max(q_next, axis=-1, keepdims=True))
                td_error = q_eval - q_target    # [H, B, 1]
                td_error = tf.reduce_sum(td_error, axis=-1)  # [H, B]

                mask_dist = tfp.distributions.Bernoulli(probs=self._probs)
                mask = tf.transpose(mask_dist.sample(batch_size), [1, 0])   # [H, B]
                q_loss = tf.reduce_mean(tf.square(td_error) * isw) + crsty_loss
            grads = tape.gradient(q_loss, self.critic_tv)
            self.optimizer.apply_gradients(
                zip(grads, self.critic_tv)
            )
            self.global_step.assign_add(1)
            return tf.reduce_mean(td_error, axis=0), dict([  # [H, B] =>
                ['LOSS/loss', q_loss],
                ['Statistics/q_max', tf.reduce_max(q_eval)],
                ['Statistics/q_min', tf.reduce_min(q_eval)],
                ['Statistics/q_mean', tf.reduce_mean(q_eval)]
            ])
