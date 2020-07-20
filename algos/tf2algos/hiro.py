import rls
import numpy as np
import tensorflow as tf
from utils.sth import sth
from tensorflow_probability import distributions as tfd
from algos.tf2algos.base.off_policy import make_off_policy_class
from utils.replay_buffer import ExperienceReplay
from rls.modules import DoubleQ


class HIRO(make_off_policy_class(mode='no_share')):
    '''
    Data-Efficient Hierarchical Reinforcement Learning, http://arxiv.org/abs/1805.08296
    '''

    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim,
                 is_continuous,

                 ployak=0.995,
                 high_scale=1.0,
                 reward_scale=1.0,
                 sample_g_nums=100,
                 sub_goal_steps=10,
                 fn_goal_dim=0,
                 intrinsic_reward_mode='os',
                 high_batch_size=256,
                 high_buffer_size=100000,
                 low_batch_size=8,
                 low_buffer_size=10000,
                 high_actor_lr=1.0e-4,
                 high_critic_lr=1.0e-3,
                 low_actor_lr=1.0e-4,
                 low_critic_lr=1.0e-3,
                 hidden_units={
                     'high_actor': [64, 64],
                     'high_critic': [64, 64],
                     'low_actor': [64, 64],
                     'low_critic': [64, 64]
                 },
                 **kwargs):
        assert visual_sources == 0, 'HIRO doesn\'t support visual inputs.'
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim=a_dim,
            is_continuous=is_continuous,
            **kwargs)
        self.data_high = ExperienceReplay(high_batch_size, high_buffer_size)
        self.data_low = ExperienceReplay(low_batch_size, low_buffer_size)

        self.ployak = ployak
        self.high_scale = np.array(
            high_scale if isinstance(high_scale, list) else [high_scale] * self.s_dim,
            dtype=np.float32)
        self.reward_scale = reward_scale
        self.fn_goal_dim = fn_goal_dim
        self.sample_g_nums = sample_g_nums
        self.sub_goal_steps = sub_goal_steps
        self.sub_goal_dim = self.s_dim - self.fn_goal_dim

        self.high_noise = rls.ClippedNormalActionNoise(mu=np.zeros(self.sub_goal_dim), sigma=self.high_scale * np.ones(self.sub_goal_dim), bound=self.high_scale / 2)
        self.low_noise = rls.ClippedNormalActionNoise(mu=np.zeros(self.a_dim), sigma=1.0 * np.ones(self.a_dim), bound=0.5)

        def _high_actor_net(): return rls.actor_dpg(self.s_dim, self.sub_goal_dim, hidden_units['high_actor'])
        if self.is_continuous:
            def _low_actor_net(): return rls.actor_dpg(self.s_dim + self.sub_goal_dim, self.a_dim, hidden_units['low_actor'])
        else:
            def _low_actor_net(): return rls.actor_discrete(self.s_dim + self.sub_goal_dim, self.a_dim, hidden_units['low_actor'])
            self.gumbel_dist = tfd.Gumbel(0, 1)

        self.high_actor = _high_actor_net()
        self.high_actor_target = _high_actor_net()
        self.low_actor = _low_actor_net()
        self.low_actor_target = _low_actor_net()

        def _high_critic_net(): return rls.critic_q_one(self.s_dim, self.sub_goal_dim, hidden_units['high_critic'])
        def _low_critic_net(): return rls.critic_q_one(self.s_dim + self.sub_goal_dim, self.a_dim, hidden_units['low_critic'])

        self.high_critic = DoubleQ(_high_critic_net)
        self.high_critic_target = DoubleQ(_high_critic_net)
        self.low_critic = DoubleQ(_low_critic_net)
        self.low_critic_target = DoubleQ(_low_critic_net)

        self.update_target_net_weights(
            self.low_actor_target.weights + self.low_critic_target.weights + self.high_actor_target.weights + self.high_critic_target.weights,
            self.low_actor.weights + self.low_critic.weights + self.high_actor.weights + self.high_critic.weights
        )

        self.low_actor_lr, self.low_critic_lr = map(self.init_lr, [low_actor_lr, low_critic_lr])
        self.high_actor_lr, self.high_critic_lr = map(self.init_lr, [high_actor_lr, high_critic_lr])
        self.low_actor_optimizer, self.low_critic_optimizer = map(self.init_optimizer, [self.low_actor_lr, self.low_critic_lr])
        self.high_actor_optimizer, self.high_critic_optimizer = map(self.init_optimizer, [self.high_actor_lr, self.high_critic_lr])

        self.model_recorder(dict(
            high_actor=self.high_actor,
            high_critic=self.high_critic,
            low_actor=self.low_actor,
            low_critic=self.low_critic,
            low_actor_optimizer=self.low_actor_optimizer,
            low_critic_optimizer=self.low_critic_optimizer,
            high_actor_optimizer=self.high_actor_optimizer,
            high_critic_optimizer=self.high_critic_optimizer
        ))

        self.counts = 0
        self._high_s = [[] for _ in range(self.n_agents)]
        self._noop_subgoal = np.random.uniform(-self.high_scale, self.high_scale, size=(self.n_agents, self.sub_goal_dim))
        self.get_ir = self.generate_ir_func(mode=intrinsic_reward_mode)

    def generate_ir_func(self, mode='os'):
        if mode == 'os':
            return lambda last_feat, subgoal, feat: -tf.norm(last_feat + subgoal - feat, ord=2, axis=-1, keepdims=True)
        elif mode == 'cos':
            return lambda last_feat, subgoal, feat: tf.expand_dims(
                -tf.keras.losses.cosine_similarity(
                    tf.cast(feat - last_feat, tf.float32),
                    tf.cast(subgoal, tf.float32),
                    axis=-1), axis=-1)

    def show_logo(self):
        self.recorder.logger.info('''
　　ｘｘｘｘｘ　ｘｘｘｘｘ　　　　　　　　ｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘｘ　　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　　　　ｘｘ　　　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘ　ｘｘｘｘ　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　　　　ｘｘ　　　　　　　　　　ｘｘ　　ｘｘｘ　　　　　　　ｘｘｘ　　　ｘｘｘ　　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　　　　ｘｘ　　　　　　　　　　ｘｘ　　ｘｘｘ　　　　　　　ｘｘ　　　　　ｘｘｘ　　
　　　　ｘｘｘｘｘｘｘ　　　　　　　　　　　ｘｘ　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　ｘｘ　　　　　ｘｘｘ　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　　　　ｘｘ　　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　ｘｘ　　　　　ｘｘｘ　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　　　　ｘｘ　　　　　　　　　　ｘｘ　ｘｘｘｘ　　　　　　　ｘｘ　　　　　ｘｘｘ　　
　　　　ｘｘ　　　ｘｘ　　　　　　　　　　　ｘｘ　　　　　　　　　　ｘｘ　　ｘｘｘ　　　　　　　ｘｘｘ　　　ｘｘｘ　　　
　　ｘｘｘｘｘ　ｘｘｘｘｘ　　　　　　　　ｘｘｘｘ　　　　　　　　ｘｘｘｘｘ　ｘｘｘｘ　　　　　　ｘｘｘｘｘｘｘ　　　
        ''')

    def store_high_buffer(self, i):
        eps_len = len(self._high_s[i])
        intervals = list(range(0, eps_len, self.sub_goal_steps))
        if len(intervals) < 1:
            return
        left = intervals[:-1]
        right = intervals[1:]
        s, r, a, g, d, s_ = [], [], [], [], [], []
        for _l, _r in zip(left, right):
            s.append(self._high_s[i][_l:_r])
            r.append(sum(self._high_r[i][_l:_r]) * self.reward_scale)
            a.append(self._high_a[i][_l:_r])
            g.append(self._subgoals[i][_l])
            d.append(self._done[i][_r - 1])
            s_.append(self._high_s_[i][_r - 1])

        right = intervals[-1]
        s.append(self._high_s[i][right:eps_len] + [self._high_s[i][-1]] * (self.sub_goal_steps + right - eps_len))
        r.append(sum(self._high_r[i][right:eps_len]))
        a.append(self._high_a[i][right:eps_len] + [self._high_a[i][-1]] * (self.sub_goal_steps + right - eps_len))
        g.append(self._subgoals[i][right])
        d.append(self._done[i][-1])
        s_.append(self._high_s_[i][-1])
        self.data_high.add(
            np.array(s),
            np.array(r)[:, np.newaxis],
            np.array(a),
            np.array(g),
            np.array(d)[:, np.newaxis],
            np.array(s_)
        )

    def reset(self):
        self._c = np.full((self.n_agents, 1), self.sub_goal_steps, np.int32)

        for i in range(self.n_agents):
            self.store_high_buffer(i)
        self._high_r = [[] for _ in range(self.n_agents)]
        self._high_a = [[] for _ in range(self.n_agents)]
        self._high_s = [[] for _ in range(self.n_agents)]
        self._subgoals = [[] for _ in range(self.n_agents)]
        self._done = [[] for _ in range(self.n_agents)]
        self._high_s_ = [[] for _ in range(self.n_agents)]

        self._new_subgoal = np.zeros((self.n_agents, self.sub_goal_dim), dtype=np.float32)

    def partial_reset(self, done):
        self._c = np.where(done[:, np.newaxis], np.full((self.n_agents, 1), self.sub_goal_steps, np.int32), self._c)
        idx = np.where(done)[0]
        for i in idx:
            self.store_high_buffer(i)
            self._high_s[i] = []
            self._high_a[i] = []
            self._high_s_[i] = []
            self._high_r[i] = []
            self._done[i] = []
            self._subgoals[i] = []

    @tf.function
    def _get_action(self, s, visual_s, subgoal):
        with tf.device(self.device):
            feat = tf.concat([s, subgoal], axis=-1)
            if self.is_continuous:
                mu = self.low_actor(feat)
                pi = tf.clip_by_value(mu + self.low_noise(), -1, 1)
            else:
                logits = self.low_actor(feat)
                mu = tf.argmax(logits, axis=1)
                cate_dist = tfd.Categorical(logits)
                pi = cate_dist.sample()
            return mu, pi

    def choose_action(self, s, visual_s, evaluation=False):
        self._subgoal = np.where(self._c == self.sub_goal_steps, self.get_subgoal(s).numpy(), self._new_subgoal)
        mu, pi = self._get_action(s, visual_s, self._subgoal)
        a = mu.numpy() if evaluation else pi.numpy()
        return a

    @tf.function
    def get_subgoal(self, s):
        '''
        last_s 上一个隐状态
        subgoal 上一个子目标
        s 当前隐状态
        '''
        new_subgoal = self.high_scale * self.high_actor(s)
        new_subgoal = tf.clip_by_value(new_subgoal + self.high_noise(), -self.high_scale, self.high_scale)
        return new_subgoal

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            if self.data_low.is_lg_batch_size and self.data_high.is_lg_batch_size:
                self.intermediate_variable_reset()
                low_data = self.get_transitions(self.data_low, data_name_list=['s', 'a', 'r', 's_', 'done', 'g', 'g_'])
                high_data = self.get_transitions(self.data_high, data_name_list=['s', 'r', 'a', 'g', 'done', 's_'])

                # --------------------------------------获取需要传给train函数的参数
                _low_training_data = self.get_value_from_dict(data_name_list=['s', 'a', 'r', 's_', 'done', 'g', 'g_'], data_dict=low_data)
                _high_training_data = self.get_value_from_dict(data_name_list=['s', 'r', 'a', 'g', 'done', 's_'], data_dict=high_data)
                summaries = self.train_low(_low_training_data)

                self.summaries.update(summaries)
                self.update_target_net_weights(self.low_actor_target.weights + self.low_critic_target.weights,
                                               self.low_actor.weights + self.low_critic.weights,
                                               self.ployak)
                if self.counts % self.sub_goal_steps == 0:
                    self.counts = 0
                    high_summaries = self.train_high(_high_training_data)
                    self.summaries.update(high_summaries)
                    self.update_target_net_weights(self.high_actor_target.weights + self.high_critic_target.weights,
                                                   self.high_actor.weights + self.high_critic.weights,
                                                   self.ployak)
                self.counts += 1
                self.summaries.update(dict([
                    ['LEARNING_RATE/low_actor_lr', self.low_actor_lr(self.train_step)],
                    ['LEARNING_RATE/low_critic_lr', self.low_critic_lr(self.train_step)],
                    ['LEARNING_RATE/high_actor_lr', self.high_actor_lr(self.train_step)],
                    ['LEARNING_RATE/high_critic_lr', self.high_critic_lr(self.train_step)]
                ]))
                self.write_training_summaries(self.global_step, self.summaries)

    @tf.function(experimental_relax_shapes=True)
    def train_low(self, memories):
        s, a, r, s_, done, g, g_ = memories
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                feat = tf.concat([s, g], axis=-1)
                feat_ = tf.concat([s_, g_], axis=-1)

                if self.is_continuous:
                    target_mu = self.low_actor_target(feat_)
                    action_target = tf.clip_by_value(target_mu + self.low_noise(), -1, 1)
                else:
                    target_logits = self.low_actor_target(feat_)
                    logp_all = tf.nn.log_softmax(target_logits)
                    gumbel_noise = tf.cast(self.gumbel_dist.sample([tf.shape(feat_)[0], self.a_dim]), dtype=tf.float32)
                    _pi = tf.nn.softmax((logp_all + gumbel_noise) / 1.)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.a_dim)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    action_target = _pi_diff + _pi
                q1, q2 = self.low_critic(feat, a)
                q = tf.minimum(q1, q2)
                q_target = self.low_critic_target.get_min(feat_, action_target)
                dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                td_error1 = q1 - dc_r
                td_error2 = q2 - dc_r
                q1_loss = tf.reduce_mean(tf.square(td_error1))
                q2_loss = tf.reduce_mean(tf.square(td_error2))
                low_critic_loss = q1_loss + q2_loss
            low_critic_grads = tape.gradient(low_critic_loss, self.low_critic.weights)
            self.low_critic_optimizer.apply_gradients(
                zip(low_critic_grads, self.low_critic.weights)
            )
            with tf.GradientTape() as tape:
                if self.is_continuous:
                    mu = self.low_actor(feat)
                else:
                    logits = self.low_actor(feat)
                    _pi = tf.nn.softmax(logits)
                    _pi_true_one_hot = tf.one_hot(tf.argmax(logits, axis=-1), self.a_dim, dtype=tf.float32)
                    _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                    mu = _pi_diff + _pi
                q_actor = self.low_critic.Q1(feat, mu)
                low_actor_loss = -tf.reduce_mean(q_actor)
            low_actor_grads = tape.gradient(low_actor_loss, self.low_actor.trainable_variables)
            self.low_actor_optimizer.apply_gradients(
                zip(low_actor_grads, self.low_actor.trainable_variables)
            )

            self.global_step.assign_add(1)
            return dict([
                ['LOSS/low_actor_loss', low_actor_loss],
                ['LOSS/low_critic_loss', low_critic_loss],
                ['Statistics/low_q_min', tf.reduce_min(q)],
                ['Statistics/low_q_mean', tf.reduce_mean(q)],
                ['Statistics/low_q_max', tf.reduce_max(q)]
            ])

    @tf.function(experimental_relax_shapes=True)
    def train_high(self, memories):
        # s_ : [B, N]
        ss, r, aa, g, done, s_ = memories

        batchs = tf.shape(ss)[0]
        # ss, aa [B, T, *]
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                s = ss[:, 0]                                # [B, N]
                true_end = (s_ - s)[:, self.fn_goal_dim:]
                g_dist = tfd.Normal(loc=true_end, scale=0.5 * self.high_scale[None, :])
                ss = tf.expand_dims(ss, 0)  # [1, B, T, *]
                ss = tf.tile(ss, [self.sample_g_nums, 1, 1, 1])    # [10, B, T, *]
                ss = tf.reshape(ss, [-1, tf.shape(ss)[-1]])  # [10*B*T, *]
                aa = tf.expand_dims(aa, 0)  # [1, B, T, *]
                aa = tf.tile(aa, [self.sample_g_nums, 1, 1, 1])    # [10, B, T, *]
                aa = tf.reshape(aa, [-1, tf.shape(aa)[-1]])  # [10*B*T, *]
                gs = tf.concat([
                    tf.expand_dims(g, 0),
                    tf.expand_dims(true_end, 0),
                    tf.clip_by_value(g_dist.sample(self.sample_g_nums - 2), -self.high_scale, self.high_scale)
                ], axis=0)  # [10, B, N]

                all_g = gs + s[:, self.fn_goal_dim:]
                all_g = tf.expand_dims(all_g, 2)    # [10, B, 1, N]
                all_g = tf.tile(all_g, [1, 1, self.sub_goal_steps, 1])   # [10, B, T, N]
                all_g = tf.reshape(all_g, [-1, tf.shape(all_g)[-1]])    # [10*B*T, N]
                all_g = all_g - ss[:, self.fn_goal_dim:]  # [10*B*T, N]
                feat = tf.concat([ss, all_g], axis=-1)  # [10*B*T, *]
                _aa = self.low_actor(feat)  # [10*B*T, A]
                if not self.is_continuous:
                    _aa = tf.one_hot(tf.argmax(_aa, axis=-1), self.a_dim, dtype=tf.float32)
                diff = _aa - aa
                diff = tf.reshape(diff, [self.sample_g_nums, batchs, self.sub_goal_steps, -1])  # [10, B, T, A]
                diff = tf.transpose(diff, [1, 0, 2, 3])   # [B, 10, T, A]
                logps = -0.5 * tf.reduce_sum(tf.norm(diff, ord=2, axis=-1)**2, axis=-1)   # [B, 10]
                idx = tf.argmax(logps, axis=-1, output_type=tf.int32)
                idx = tf.stack([tf.range(batchs), idx], axis=1)  # [B, 2]
                g = tf.gather_nd(tf.transpose(gs, [1, 0, 2]), idx)  # [B, N]

                q1, q2 = self.high_critic(s, g)
                q = tf.minimum(q1, q2)

                target_sub_goal = self.high_actor_target(s_) * self.high_scale
                target_sub_goal = tf.clip_by_value(target_sub_goal + self.high_noise(), -self.high_scale, self.high_scale)
                q_target = self.high_critic_target.get_min(s_, target_sub_goal)

                dc_r = tf.stop_gradient(r + self.gamma * (1 - done) * q_target)
                td_error1 = q1 - dc_r
                td_error2 = q2 - dc_r
                q1_loss = tf.reduce_mean(tf.square(td_error1))
                q2_loss = tf.reduce_mean(tf.square(td_error2))
                high_critic_loss = q1_loss + q2_loss

            high_critic_grads = tape.gradient(high_critic_loss, self.high_critic.weights)
            self.high_critic_optimizer.apply_gradients(
                zip(high_critic_grads, self.high_critic.weights)
            )
            with tf.GradientTape() as tape:
                mu = self.high_actor(s) * self.high_scale
                q_actor = self.high_critic.Q1(s, mu)
                high_actor_loss = -tf.reduce_mean(q_actor)
            high_actor_grads = tape.gradient(high_actor_loss, self.high_actor.trainable_variables)
            self.high_actor_optimizer.apply_gradients(
                zip(high_actor_grads, self.high_actor.trainable_variables)
            )
            return dict([
                ['LOSS/high_actor_loss', high_actor_loss],
                ['LOSS/high_critic_loss', high_critic_loss],
                ['Statistics/high_q_min', tf.reduce_min(q)],
                ['Statistics/high_q_mean', tf.reduce_mean(q)],
                ['Statistics/high_q_max', tf.reduce_max(q)]
            ])

    def no_op_store(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store need done type is np.ndarray"
        [o.append(_s) for o, _s in zip(self._high_s, s)]
        [o.append(_a) for o, _a in zip(self._high_a, a)]
        [o.append(_r) for o, _r in zip(self._high_r, r)]
        [o.append(_s_) for o, _s_ in zip(self._high_s_, s_)]
        [o.append(_d) for o, _d in zip(self._done, done)]
        [o.append(_subgoal) for o, _subgoal in zip(self._subgoals, self._noop_subgoal)]

        ir = self.get_ir(s[:, self.fn_goal_dim:], self._noop_subgoal, s_[:, self.fn_goal_dim:])
        # subgoal = s[:, self.fn_goal_dim:] + self._noop_subgoal - s_[:, self.fn_goal_dim:]
        subgoal = np.random.uniform(-self.high_scale, self.high_scale, size=(self.n_agents, self.sub_goal_dim))
        self.data_low.add(
            s,
            a,
            ir,
            s_,
            done[:, np.newaxis],  # 升维
            self._noop_subgoal,
            subgoal
        )
        self._noop_subgoal = subgoal

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        """
        for off-policy training, use this function to store <s, a, r, s_, done> into ReplayBuffer.
        """
        assert isinstance(a, np.ndarray), "store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store need done type is np.ndarray"
        [o.append(_s) for o, _s in zip(self._high_s, s)]
        [o.append(_a) for o, _a in zip(self._high_a, a)]
        [o.append(_r) for o, _r in zip(self._high_r, r)]
        [o.append(_s_) for o, _s_ in zip(self._high_s_, s_)]
        [o.append(_d) for o, _d in zip(self._done, done)]
        [o.append(_subgoal) for o, _subgoal in zip(self._subgoals, self._subgoal)]

        ir = self.get_ir(s[:, self.fn_goal_dim:], self._subgoal, s_[:, self.fn_goal_dim:])
        self._new_subgoal = np.where(self._c == 1, self.get_subgoal(s_).numpy(), s[:, self.fn_goal_dim:] + self._subgoal - s_[:, self.fn_goal_dim:])

        self.data_low.add(
            s,
            a,
            ir,
            s_,
            done[:, np.newaxis],  # 升维
            self._subgoal,
            self._new_subgoal
        )
        self._c = np.where(self._c == 1, np.full((self.n_agents, 1), self.sub_goal_steps, np.int32), self._c - 1)

    def get_transitions(self, databuffer, data_name_list=['s', 'a', 'r', 's_', 'done']):
        '''
        TODO: Annotation
        '''
        data = databuffer.sample()   # 经验池取数据
        if not self.is_continuous and 'a' in data_name_list:
            a_idx = data_name_list.index('a')
            a = data[a_idx].astype(np.int32)
            pre_shape = a.shape
            a = a.reshape(-1)
            a = sth.int2one_hot(a, self.a_dim)
            a = a.reshape(pre_shape + (-1,))
            data[a_idx] = a
        return dict([
            [n, d] for n, d in zip(data_name_list, list(map(self.data_convert, data)))
        ])
