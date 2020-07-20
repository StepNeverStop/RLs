import rls
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from algos.tf2algos.base.off_policy import make_off_policy_class
from utils.expl_expt import ExplorationExploitationClass
from rls.modules import DoubleQ


class MAXSQN(make_off_policy_class(mode='share')):
    '''
    https://github.com/createamind/DRL/blob/master/spinup/algos/maxsqn/maxsqn.py
    '''

    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim,
                 is_continuous,

                 alpha=0.2,
                 beta=0.1,
                 ployak=0.995,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 use_epsilon=False,
                 q_lr=5.0e-4,
                 alpha_lr=5.0e-4,
                 auto_adaption=True,
                 hidden_units=[32, 32],
                 **kwargs):
        assert not is_continuous, 'maxsqn only support discrete action space'
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
        self.use_epsilon = use_epsilon
        self.ployak = ployak
        self.log_alpha = alpha if not auto_adaption else tf.Variable(initial_value=0.0, name='log_alpha', dtype=tf.float32, trainable=True)
        self.auto_adaption = auto_adaption
        self.target_entropy = beta * np.log(self.a_dim)

        def _q_net(): return rls.critic_q_all(self.feat_dim, self.a_dim, hidden_units)
        self.critic_net = DoubleQ(_q_net)
        self.critic_target_net = DoubleQ(_q_net)
        self.critic_tv = self.critic_net.trainable_variables + self.other_tv
        self.update_target_net_weights(self.critic_target_net.weights, self.critic_net.weights)
        self.q_lr, self.alpha_lr = map(self.init_lr, [q_lr, alpha_lr])
        self.optimizer_critic, self.optimizer_alpha = map(self.init_optimizer, [self.q_lr, self.alpha_lr])

        self.model_recorder(dict(
            critic_net=self.critic_net,
            optimizer_critic=self.optimizer_critic,
            optimizer_alpha=self.optimizer_alpha
        ))

    def show_logo(self):
        self.recorder.logger.info('''
　　　ｘｘ　　　　　ｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘｘｘｘ　　　ｘｘ　　　
　　　ｘｘｘ　　　ｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘ　ｘｘｘ　　　　　　　　ｘｘｘｘ　ｘｘｘ　　　　　　ｘｘｘｘ　　　ｘｘ　　　
　　　ｘｘｘ　　　ｘｘｘ　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　ｘｘ　　　　　　　　　　　　　ｘｘ　　　　ｘｘ　　　　　　ｘｘｘｘｘ　　ｘｘ　　　
　　　ｘｘｘｘ　　ｘｘｘ　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　ｘｘ　ｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘｘ　　　　ｘｘｘ　　　　　ｘｘ　ｘｘｘ　ｘｘ　　　
　　　ｘｘｘｘ　ｘｘ　ｘ　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘｘ　　　　　　ｘｘ　　　　　ｘｘ　　ｘｘｘｘｘ　　　
　　　ｘｘｘｘ　ｘｘ　ｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘ　　　　　　　　　　　　　　　ｘｘｘ　　　　　　ｘｘｘ　　ｘ　ｘｘｘ　　　　　ｘｘ　　　ｘｘｘｘ　　　
　　　ｘｘ　ｘｘｘ　　ｘ　　　　　　　ｘｘｘ　　ｘｘ　　　　　　　　　　ｘｘｘ　　　　　　　　　　ｘｘ　　　　ｘｘ　　　　　　　ｘｘ　ｘｘｘｘｘ　　　　　　ｘｘ　　　ｘｘｘｘ　　　
　　　ｘｘ　ｘｘｘ　　ｘ　　　　　　　ｘｘ　　ｘｘｘ　　　　　　　　　ｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘ　ｘｘｘｘ　　　　　　ｘｘ　　　　ｘｘｘ　　　
　　　ｘｘ　ｘｘｘ　　ｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘ　ｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　ｘｘ　　　　　ｘｘ　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　　　　　　　　　　　　　　
        ''')

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    def choose_action(self, s, visual_s, evaluation=False):
        if self.use_epsilon and np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
            a = np.random.randint(0, self.a_dim, self.n_agents)
        else:
            mu, pi, self.cell_state = self._get_action(s, visual_s, self.cell_state)
            a = pi.numpy()
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state):
        with tf.device(self.device):
            feat, cell_state = self.get_feature(s, visual_s, cell_state=cell_state, record_cs=True)
            q = self.critic_net.Q1(feat)
            cate_dist = tfp.distributions.Categorical(logits=q / self.alpha)
            pi = cate_dist.sample()
        return tf.argmax(q, axis=1), pi, cell_state

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'train_function': self.train,
                'update_function': lambda: self.update_target_net_weights(self.critic_target_net.weights, self.critic_net.weights,
                                                                          self.ployak),
                'summary_dict': dict([
                    ['LEARNING_RATE/q_lr', self.q_lr(self.train_step)],
                    ['LEARNING_RATE/alpha_lr', self.alpha_lr(self.train_step)]
                ])
            })

    @tf.function(experimental_relax_shapes=True)
    def train(self, memories, isw, crsty_loss, cell_state):
        ss, vvss, a, r, done = memories
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                q1, q2 = self.critic_net(feat)
                q1_eval = tf.reduce_sum(tf.multiply(q1, a), axis=1, keepdims=True)
                q2_eval = tf.reduce_sum(tf.multiply(q2, a), axis=1, keepdims=True)

                q1_target, q2_target = self.critic_target_net(feat_)
                q1_target_max = tf.reduce_max(q1_target, axis=1, keepdims=True)
                q1_target_log_probs = tf.nn.log_softmax(q1_target / self.alpha, axis=1) + 1e-8
                q1_target_entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(q1_target_log_probs) * q1_target_log_probs, axis=1, keepdims=True))

                q2_target_max = tf.reduce_max(q2_target, axis=1, keepdims=True)
                # q2_target_log_probs = tf.nn.log_softmax(q2_target, axis=1)
                # q2_target_log_max = tf.reduce_max(q2_target_log_probs, axis=1, keepdims=True)

                q_target = tf.minimum(q1_target_max, q2_target_max) + self.alpha * q1_target_entropy
                dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                td_error1 = q1_eval - dc_r
                td_error2 = q2_eval - dc_r
                q1_loss = tf.reduce_mean(tf.square(td_error1) * isw)
                q2_loss = tf.reduce_mean(tf.square(td_error2) * isw)
                loss = 0.5 * (q1_loss + q2_loss) + crsty_loss
            loss_grads = tape.gradient(loss, self.critic_tv)
            self.optimizer_critic.apply_gradients(
                zip(loss_grads, self.critic_tv)
            )
            if self.auto_adaption:
                with tf.GradientTape() as tape:
                    q1 = self.critic_net.Q1(feat)
                    q1_log_probs = tf.nn.log_softmax(q1 / self.alpha, axis=1) + 1e-8
                    q1_entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(q1_log_probs) * q1_log_probs, axis=1, keepdims=True))
                    alpha_loss = -tf.reduce_mean(self.alpha * tf.stop_gradient(self.target_entropy - q1_entropy))
                alpha_grad = tape.gradient(alpha_loss, self.log_alpha)
                self.optimizer_alpha.apply_gradients(
                    [(alpha_grad, self.log_alpha)]
                )
            self.global_step.assign_add(1)
            summaries = dict([
                ['LOSS/loss', loss],
                ['Statistics/log_alpha', self.log_alpha],
                ['Statistics/alpha', self.alpha],
                ['Statistics/q1_entropy', q1_entropy],
                ['Statistics/q_min', tf.reduce_mean(tf.minimum(q1, q2))],
                ['Statistics/q_mean', tf.reduce_mean(q1)],
                ['Statistics/q_max', tf.reduce_mean(tf.maximum(q1, q2))]
            ])
            if self.auto_adaption:
                summaries.update({
                    'LOSS/alpha_loss': alpha_loss
                })
            return (td_error1 + td_error2) / 2, summaries
