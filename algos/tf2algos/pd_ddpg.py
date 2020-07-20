import rls
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from algos.tf2algos.base.off_policy import make_off_policy_class


class PD_DDPG(make_off_policy_class(mode='share')):
    '''
    Accelerated Primal-Dual Policy Optimization for Safe Reinforcement Learning, http://arxiv.org/abs/1802.06480
    Refer to https://github.com/anita-hu/TF2-RL/blob/master/Primal-Dual_DDPG/TF2_PD_DDPG_Basic.py
    '''

    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim,
                 is_continuous,

                 ployak=0.995,
                 actor_lr=5.0e-4,
                 reward_critic_lr=1.0e-3,
                 cost_critic_lr=1.0e-3,
                 lambda_lr=5.0e-4,
                 discrete_tau=1.0,
                 cost_constraint=1.0,
                 hidden_units={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'reward': [32, 32],
                     'cost': [32, 32]
                 },
                 **kwargs):
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim=a_dim,
            is_continuous=is_continuous,
            **kwargs)
        self.ployak = ployak
        self.discrete_tau = discrete_tau
        self._lambda = tf.Variable(0.0, dtype=tf.float32)
        self.cost_constraint = cost_constraint  # long tern cost <= d

        if self.is_continuous:
            def _actor_net(): return rls.actor_dpg(self.feat_dim, self.a_dim, hidden_units['actor_continuous'])
            # self.action_noise = rls.NormalActionNoise(mu=np.zeros(self.a_dim), sigma=1 * np.ones(self.a_dim))
            self.action_noise = rls.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_dim), sigma=0.2 * np.ones(self.a_dim))
        else:
            def _actor_net(): return rls.actor_discrete(self.feat_dim, self.a_dim, hidden_units['actor_discrete'])
            self.gumbel_dist = tfp.distributions.Gumbel(0, 1)

        self.actor_net = _actor_net()
        self.actor_target_net = _actor_net()
        self.actor_tv = self.actor_net.trainable_variables

        def _critic_net(hiddens): return rls.critic_q_one(self.feat_dim, self.a_dim, hiddens)
        self.reward_critic_net = _critic_net(hidden_units['reward'])
        self.reward_critic_target_net = _critic_net(hidden_units['reward'])
        self.cost_critic_net = _critic_net(hidden_units['cost'])
        self.cost_critic_target_net = _critic_net(hidden_units['cost'])

        self.reward_critic_tv = self.reward_critic_net.trainable_variables + self.other_tv
        self.update_target_net_weights(
            self.actor_target_net.weights + self.reward_critic_target_net.weights + self.cost_critic_target_net.weights,
            self.actor_net.weights + self.reward_critic_net.weights + self.cost_critic_net.weights
        )
        self.lambda_lr = lambda_lr
        self.actor_lr, self.reward_critic_lr, self.cost_critic_lr = map(self.init_lr, [actor_lr, reward_critic_lr, cost_critic_lr])
        self.optimizer_actor, self.optimizer_reward_critic, self.optimizer_cost_critic = map(self.init_optimizer, [self.actor_lr, self.reward_critic_lr, self.cost_critic_lr])

        self.model_recorder(dict(
            actor=self.actor_net,
            reward_critic=self.reward_critic_net,
            cost_critic=self.cost_critic_net,
            optimizer_actor=self.optimizer_actor,
            optimizer_reward_critic=self.optimizer_reward_critic,
            optimizer_cost_critic=self.optimizer_cost_critic
        ))

    def show_logo(self):
        self.recorder.logger.info('''
　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　　　ｘｘ　　ｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　　　　　　　
　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘ　　　ｘｘｘｘｘ　　　
　　　　　ｘ　　　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　ｘｘｘ　　　　
　　　　　ｘ　　　　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　　ｘ　　　　　　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　　ｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　　　
        ''')

    def choose_action(self, s, visual_s, evaluation=False):
        mu, pi, self.cell_state = self._get_action(s, visual_s, self.cell_state)
        a = mu.numpy() if evaluation else pi.numpy()
        return a

    @tf.function
    def _get_action(self, s, visual_s, cell_state):
        with tf.device(self.device):
            feat, cell_state = self.get_feature(s, visual_s, cell_state=cell_state, record_cs=True)
            if self.is_continuous:
                mu = self.actor_net(feat)
                pi = tf.clip_by_value(mu + self.action_noise(), -1, 1)
            else:
                logits = self.actor_net(feat)
                mu = tf.argmax(logits, axis=1)
                cate_dist = tfp.distributions.Categorical(logits)
                pi = cate_dist.sample()
            return mu, pi, cell_state

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'train_function': self.train,
                'update_function': lambda: self.update_target_net_weights(
                    self.actor_target_net.weights + self.reward_critic_target_net.weights + self.cost_critic_target_net.weights,
                    self.actor_net.weights + self.reward_critic_net.weights + self.cost_critic_net.weights,
                    self.ployak),
                'summary_dict': dict([
                    ['LEARNING_RATE/actor_lr', self.actor_lr(self.train_step)],
                    ['LEARNING_RATE/reward_critic_lr', self.reward_critic_lr(self.train_step)],
                    ['LEARNING_RATE/cost_critic_lr', self.cost_critic_lr(self.train_step)]
                ]),
                'sample_data_list': ['s', 'visual_s', 'a', 'r', 's_', 'visual_s_', 'done', 'cost'],
                'train_data_list': ['ss', 'vvss', 'a', 'r', 'done', 'cost'],
            })

    @tf.function(experimental_relax_shapes=True)
    def train(self, memories, isw, crsty_loss, cell_state):
        ss, vvss, a, r, done, cost = memories
        batch_size = tf.shape(a)[0]
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                if self.is_continuous:
                    target_mu = self.actor_target_net(feat_)
                    action_target = tf.clip_by_value(target_mu + self.action_noise(), -1, 1)
                else:
                    target_logits = self.actor_target_net(feat_)
                    logp_all = tf.nn.log_softmax(target_logits)
                    gumbel_noise = tf.cast(self.gumbel_dist.sample([batch_size, self.a_dim]), dtype=tf.float32)
                    _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_dim)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    action_target = _pi_diff + _pi
                q_reward = self.reward_critic_net(feat, a)
                q_target = self.reward_critic_target_net(feat_, action_target)
                dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                td_error_reward = q_reward - dc_r
                reward_loss = 0.5 * tf.reduce_mean(tf.square(td_error_reward) * isw) + crsty_loss
            q_grads = tape.gradient(reward_loss, self.reward_critic_tv)
            self.optimizer_reward_critic.apply_gradients(
                zip(q_grads, self.reward_critic_tv)
            )

            with tf.GradientTape() as tape:
                q_cost = self.cost_critic_net(feat, a)
                q_target = self.cost_critic_target_net(feat_, action_target)
                dc_r = tf.stop_gradient(cost + self.gamma * q_target * (1 - done))
                td_error_cost = q_cost - dc_r
                cost_loss = 0.5 * tf.reduce_mean(tf.square(td_error_cost) * isw) + crsty_loss
            q_grads = tape.gradient(cost_loss, self.cost_critic_net.trainable_variables)
            self.optimizer_cost_critic.apply_gradients(
                zip(q_grads, self.cost_critic_net.trainable_variables)
            )

            q_loss = reward_loss + cost_loss

            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu = self.actor_net(feat)
                else:
                    logits = self.actor_net(feat)
                    _pi = tf.nn.softmax(logits)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(logits, axis=-1), self.a_dim, dtype=tf.float32)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    mu = _pi_diff + _pi
                reward_actor = self.reward_critic_net(feat, mu)
                cost_actor = self.cost_critic_net(feat, mu)
                actor_loss = -tf.reduce_mean(reward_actor - self._lambda * cost_actor)
            actor_grads = tape.gradient(actor_loss, self.actor_tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_tv)
            )

            # update dual variable
            lambda_update = tf.reduce_mean(cost_actor - self.cost_constraint)
            self._lambda.assign_add(self.lambda_lr * lambda_update)
            self._lambda.assign(tf.maximum(self._lambda, 0.0))

            self.global_step.assign_add(1)
            return (td_error_reward + td_error_cost) / 2, dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/reward_loss', reward_loss],
                ['LOSS/cost_loss', cost_loss],
                ['LOSS/q_loss', q_loss],
                ['Statistics/q_reward_min', tf.reduce_min(q_reward)],
                ['Statistics/q_reward_mean', tf.reduce_mean(q_reward)],
                ['Statistics/q_reward_max', tf.reduce_max(q_reward)],
                ['Statistics/q_cost_min', tf.reduce_min(q_cost)],
                ['Statistics/q_cost_mean', tf.reduce_mean(q_cost)],
                ['Statistics/q_cost_max', tf.reduce_max(q_cost)],
                ['Statistics/_lambda', self._lambda],
                ['Statistics/lambda_update', lambda_update]
            ])

    @tf.function(experimental_relax_shapes=True)
    def train_persistent(self, memories, isw, crsty_loss, cell_state):
        ss, vvss, a, r, done = memories
        batch_size = tf.shape(a)[0]
        with tf.device(self.device):
            with tf.GradientTape(persistent=True) as tape:
                feat, feat_ = self.get_feature(ss, vvss, cell_state=cell_state, s_and_s_=True)
                if self.is_continuous:
                    target_mu = self.actor_target_net(feat_)
                    action_target = tf.clip_by_value(target_mu + self.action_noise(), -1, 1)
                    mu = self.actor_net(feat)
                else:
                    target_logits = self.actor_target_net(feat_)
                    logp_all = tf.nn.log_softmax(target_logits)
                    gumbel_noise = tf.cast(self.gumbel_dist.sample([batch_size, self.a_dim]), dtype=tf.float32)
                    _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_dim)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    action_target = _pi_diff + _pi
                    logits = self.actor_net(feat)
                    _pi = tf.nn.softmax(logits)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(logits, axis=-1), self.a_dim, dtype=tf.float32)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    mu = _pi_diff + _pi
                q_reward = self.reward_critic_net(feat, a)
                q_target = self.reward_critic_target_net(feat_, action_target)
                dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                td_error_reward = q_reward - dc_r
                reward_loss = 0.5 * tf.reduce_mean(tf.square(td_error_reward) * isw) + crsty_loss

                q_cost = self.cost_critic_net(tf.stop_gradient(feat), a)
                q_target = self.cost_critic_target_net(feat_, action_target)
                dc_r = tf.stop_gradient(cost + self.gamma * q_target * (1 - done))
                td_error_cost = q_cost - dc_r
                cost_loss = 0.5 * tf.reduce_mean(tf.square(td_error_cost) * isw) + crsty_loss

                q_loss = reward_loss + cost_loss

                reward_actor = self.reward_critic_net(feat, mu)
                cost_actor = self.cost_critic_net(feat, mu)
                actor_loss = -tf.reduce_mean(reward_actor - self._lambda * cost_actor)
            q_grads = tape.gradient(reward_loss, self.reward_critic_tv)
            self.optimizer_reward_critic.apply_gradients(
                zip(q_grads, self.reward_critic_tv)
            )
            q_grads = tape.gradient(cost_loss, self.cost_critic_net.trainable_variables)
            self.optimizer_cost_critic.apply_gradients(
                zip(q_grads, self.cost_critic_net.trainable_variables)
            )
            actor_grads = tape.gradient(actor_loss, self.actor_tv)
            self.optimizer_actor.apply_gradients(
                zip(actor_grads, self.actor_tv)
            )

            # update dual variable
            lambda_update = tf.reduce_mean(cost_actor - self.cost_constraint)
            self._lambda.assign_add(self.lambda_lr * lambda_update)
            self._lambda.assign(tf.maximum(self._lambda, 0.0))

            self.global_step.assign_add(1)
            return (td_error_reward + td_error_cost) / 2, dict([
                ['LOSS/actor_loss', actor_loss],
                ['LOSS/reward_loss', reward_loss],
                ['LOSS/cost_loss', cost_loss],
                ['LOSS/q_loss', q_loss],
                ['Statistics/q_reward_min', tf.reduce_min(q_reward)],
                ['Statistics/q_reward_mean', tf.reduce_mean(q_reward)],
                ['Statistics/q_reward_max', tf.reduce_max(q_reward)],
                ['Statistics/q_cost_min', tf.reduce_min(q_cost)],
                ['Statistics/q_cost_mean', tf.reduce_mean(q_cost)],
                ['Statistics/q_cost_max', tf.reduce_max(q_cost)],
                ['Statistics/_lambda', self._lambda],
                ['Statistics/lambda_update', lambda_update]
            ])

    def get_cost(self, s, visual_s, a, r, s_, visual_s_, done):
        return np.abs(s_)[:, :1]    # CartPole

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        """
        for off-policy training, use this function to store <s, a, r, s_, done> into ReplayBuffer.
        """
        assert isinstance(a, np.ndarray), "store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store need done type is np.ndarray"
        self._running_average(s)
        cost = self.get_cost(s, visual_s, a, r, s_, visual_s_, done)
        self.data.add(
            s,
            visual_s,
            a,
            r[:, np.newaxis],   # 升维
            s_,
            visual_s_,
            done[:, np.newaxis],  # 升维
            cost
        )

    def no_op_store(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "no_op_store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "no_op_store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "no_op_store need done type is np.ndarray"
        self._running_average(s)
        cost = self.get_cost(s, visual_s, a, r, s_, visual_s_, done)
        self.data.add(
            s,
            visual_s,
            a,
            r[:, np.newaxis],
            s_,
            visual_s_,
            done[:, np.newaxis],  # 升维
            cost
        )
