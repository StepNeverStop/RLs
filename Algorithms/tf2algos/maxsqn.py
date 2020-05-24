import Nn
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from Algorithms.tf2algos.base.off_policy import Off_Policy
from utils.expl_expt import ExplorationExploitationClass


class MAXSQN(Off_Policy):
    '''
    https://github.com/createamind/DRL/blob/master/spinup/algos/maxsqn/maxsqn.py
    '''
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,

                 alpha=0.2,
                 beta=0.1,
                 ployak=0.995,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_episode=100,
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
            a_dim_or_list=a_dim_or_list,
            is_continuous=is_continuous,
            **kwargs)
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_episode=init2mid_annealing_episode,
                                                          max_episode=self.max_episode)
        self.use_epsilon = use_epsilon
        self.ployak = ployak
        self.log_alpha = alpha if not auto_adaption else tf.Variable(initial_value=0.0, name='log_alpha', dtype=tf.float32, trainable=True)
        self.auto_adaption = auto_adaption
        self.target_alpha = beta * np.log(self.a_counts)

        _q_net = lambda : Nn.critic_q_all(self.rnn_net.hdim, self.a_counts, hidden_units)

        self.q1_net = _q_net()
        self.q1_target_net = _q_net()
        self.q2_net = _q_net()
        self.q2_target_net = _q_net()
        self.critic_tv = self.q1_net.trainable_variables + self.q2_net.trainable_variables + self.other_tv
        self.update_target_net_weights(
            self.q1_target_net.weights + self.q2_target_net.weights,
            self.q1_net.weights + self.q2_net.weights
        )
        self.q_lr, self.alpha_lr = map(self.init_lr, [q_lr, alpha_lr])
        self.optimizer_critic, self.optimizer_alpha = map(self.init_optimizer, [self.q_lr, self.alpha_lr])

        self.model_recorder(dict(
            q1_net=self.q1_net,
            q2_net=self.q2_net,
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

    def choose_action(self, s, visual_s, evaluation=False):
        if self.use_epsilon and np.random.uniform() < self.expl_expt_mng.get_esp(self.episode, evaluation=evaluation):
            a = np.random.randint(0, self.a_counts, self.n_agents)
        else:
            mu, pi, self.cell_state = self._get_action(s, visual_s, self.cell_state)
            a = pi.numpy()
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state):
        with tf.device(self.device):
            feat, cell_state = self.get_feature(s, visual_s, cell_state=cell_state, record_cs=True, train=False)
            q = self.q1_net(feat)
            cate_dist = tfp.distributions.Categorical(logits=q / tf.exp(self.log_alpha))
            pi = cate_dist.sample()
        return tf.argmax(q, axis=1), pi, cell_state

    def learn(self, **kwargs):
        self.episode = kwargs['episode']
        for i in range(kwargs['step']):
            self._learn(function_dict={
                'train_function': self.train,
                'update_function': lambda : self.update_target_net_weights(
                                            self.q1_target_net.weights + self.q2_target_net.weights,
                                            self.q1_net.weights + self.q2_net.weights,
                                            self.ployak),
                'summary_dict': dict([
                                    ['LEARNING_RATE/q_lr', self.q_lr(self.episode)],
                                    ['LEARNING_RATE/alpha_lr', self.alpha_lr(self.episode)]
                                ])
            })

    @tf.function(experimental_relax_shapes=True)
    def train(self, memories, isw, crsty_loss, cell_state):
        ss, vvss, a, r, done = memories
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                q1 = self.q1_net(feat)
                q1_eval = tf.reduce_sum(tf.multiply(q1, a), axis=1, keepdims=True)
                q2 = self.q2_net(feat)
                q2_eval = tf.reduce_sum(tf.multiply(q2, a), axis=1, keepdims=True)

                q1_target = self.q1_target_net(feat_)
                q1_target_max = tf.reduce_max(q1_target, axis=1, keepdims=True)
                q1_target_log_probs = tf.nn.log_softmax(q1_target / tf.exp(self.log_alpha), axis=1) + 1e-8
                q1_target_log_max = tf.reduce_max(q1_target_log_probs, axis=1, keepdims=True)
                q1_target_entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(q1_target_log_probs) * q1_target_log_probs, axis=1, keepdims=True))

                q2_target = self.q2_target_net(feat_)
                q2_target_max = tf.reduce_max(q2_target, axis=1, keepdims=True)
                # q2_target_log_probs = tf.nn.log_softmax(q2_target, axis=1)
                # q2_target_log_max = tf.reduce_max(q2_target_log_probs, axis=1, keepdims=True)

                q_target = tf.minimum(q1_target_max, q2_target_max) + tf.exp(self.log_alpha) * q1_target_entropy
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
                    q1 = self.q1_net(feat)
                    q1_log_probs = tf.nn.log_softmax(q1_target / tf.exp(self.log_alpha), axis=1) + 1e-8
                    q1_log_max = tf.reduce_max(q1_log_probs, axis=1, keepdims=True)
                    q1_entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(q1_log_probs) * q1_log_probs, axis=1, keepdims=True))
                    alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(self.target_alpha - q1_entropy))
                alpha_grad = tape.gradient(alpha_loss, self.log_alpha)
                self.optimizer_alpha.apply_gradients(
                    [(alpha_grad, self.log_alpha)]
                )
            self.global_step.assign_add(1)
            summaries = dict([
                ['LOSS/loss', loss],
                ['Statistics/log_alpha', self.log_alpha],
                ['Statistics/alpha', tf.exp(self.log_alpha)],
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
