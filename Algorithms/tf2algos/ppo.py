import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import Nn
from utils.sth import sth
from .policy import Policy


class PPO(Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 action_type,
                 epsilon=0.2,
                 gamma=0.99,
                 beta=1.0e-3,
                 lr=5.0e-4,
                 lambda_=0.95,
                 max_episode=50000,
                 batch_size=100,
                 epoch=5,
                 base_dir=None,
                 logger2file=False,
                 out_graph=False):
        super().__init__(s_dim, visual_sources, visual_resolution, a_dim_or_list, action_type, gamma, max_episode, base_dir, 'ON')
        self.beta = beta
        self.epoch = epoch
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.lr = lr
        self.sigma_offset = np.full([self.a_counts, ], 0.01)
        if self.action_type == 'continuous':
            self.net = Nn.a_c_v_continuous(self.s_dim, self.visual_dim, self.a_counts, 'ppo')
        else:
            self.net = Nn.a_c_v_discrete(self.s_dim, self.visual_dim, self.a_counts, 'ppo')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.generate_recorder(
            logger2file=logger2file,
            model=self
        )
        self.recorder.logger.info('''
　　　ｘｘｘｘｘｘｘｘ　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘ　　　　　
　　　　　ｘｘ　　ｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　　　　　ｘｘｘ　ｘｘｘ　　　　
　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　ｘｘ　　　ｘｘ　　　　
　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　ｘｘ　　　ｘｘｘ　　　
　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘｘｘ　　　ｘｘｘ　　　
　　　　　ｘ　　　　　　　　　　　　　　ｘ　　　　　　　　　　　　　ｘｘ　　　ｘｘｘ　　　
　　　　　ｘ　　　　　　　　　　　　　　ｘ　　　　　　　　　　　　　ｘｘ　　　ｘｘ　　　　
　　　　　ｘ　　　　　　　　　　　　　　ｘ　　　　　　　　　　　　　ｘｘ　　ｘｘｘ　　　　
　　　ｘｘｘｘｘ　　　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　　ｘｘｘｘｘ　　
        ''')

    def choose_action(self, s, visual_s):
        if self.action_type == 'continuous':
            return self._get_action(s, visual_s).numpy()
        else:
            if np.random.uniform() < self.epsilon:
                a = np.random.randint(0, self.a_counts, len(s))
            else:
                a = self._get_action(s, visual_s).numpy()
            return sth.int2action_index(a, self.a_dim_or_list)

    def choose_inference_action(self, s, visual_s):
        a = self._get_action(s, visual_s).numpy()
        return a if self.action_type == 'continuous' else sth.int2action_index(a, self.a_dim_or_list)

    @tf.function
    def _get_action(self, vector_input, visual_input):
        with tf.device(self.device):
            if self.action_type == 'continuous':
                mu, sigma, value = self.net(vector_input, visual_input)
                norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma + self.sigma_offset)
                sample_op = tf.clip_by_value(norm_dist.sample(), -1, 1)
            else:
                action_probs, value = self.net(vector_input, visual_input)
                sample_op = tf.argmax(action_probs, axis=1)
        return sample_op

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "store_data need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store_data need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store_data need done type is np.ndarray"
        if not self.action_type == 'continuous':
            a = sth.action_index2one_hot(a, self.a_dim_or_list)
        self.data = self.data.append({
            's': s,
            'visual_s': visual_s,
            'a': a,
            'r': r,
            's_': s_,
            'visual_s_': visual_s_,
            'done': done,
            'value': np.squeeze(self._get_value(s, visual_s).numpy()),
            'next_value': np.squeeze(self._get_value(s_, visual_s_).numpy()),
            'prob': self._get_new_prob(s, visual_s, a).numpy() + 1e-10
        }, ignore_index=True)

    @tf.function
    def _get_value(self, s, visual_s):
        with tf.device(self.device):
            if self.action_type == 'continuous':
                mu, sigma, value = self.net(s, visual_s)
            else:
                action_probs, value = self.net(s, visual_s)
            return value

    @tf.function
    def _get_new_prob(self, s, visual_s, a):
        with tf.device(self.device):
            if self.action_type == 'continuous':
                mu, sigma, value = self.net(s, visual_s)
                norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma + self.sigma_offset)
                new_prob = tf.reduce_mean(norm_dist.prob(a), axis=1, keepdims=True)
            else:
                action_probs, value = self.net(s, visual_s)
                new_prob = tf.reduce_max(action_probs, axis=1, keepdims=True)
            return new_prob

    def calculate_statistics(self):
        self.data['total_reward'] = sth.discounted_sum(self.data.r.values, 1, 0, self.data.done.values)
        self.data['discounted_reward'] = sth.discounted_sum(self.data.r.values, self.gamma, self.data.next_value.values[-1], self.data.done.values)
        self.data['td_error'] = sth.discounted_sum_minus(
            self.data.r.values,
            self.gamma,
            self.data.next_value.values[-1],
            self.data.done.values,
            self.data.value.values
        )
        # GAE
        self.data['advantage'] = sth.discounted_sum(
            self.data.td_error.values,
            self.lambda_ * self.gamma,
            0,
            self.data.done.values
        )
        # self.data.to_excel(self.excel_writer, sheet_name='test', index=True)
        # self.excel_writer.save()

    def get_sample_data(self):
        i_data = self.data.sample(n=self.batch_size) if self.batch_size < self.data.shape[0] else self.data
        s = np.vstack([i_data.s.values[i] for i in range(i_data.shape[0])])
        visual_s = np.vstack([i_data.visual_s.values[i] for i in range(i_data.shape[0])])
        a = np.vstack([i_data.a.values[i] for i in range(i_data.shape[0])])
        dc_r = np.vstack([i_data.discounted_reward.values[i][:, np.newaxis] for i in range(i_data.shape[0])])
        old_prob = np.vstack([i_data.prob.values[i] for i in range(i_data.shape[0])])
        advantage = np.vstack([i_data.advantage.values[i][:, np.newaxis] for i in range(i_data.shape[0])])
        return s, visual_s, a, dc_r, old_prob, advantage

    def learn(self, episode):
        self.calculate_statistics()
        for _ in range(self.epoch):
            s, visual_s, a, dc_r, old_prob, advantage = self.get_sample_data()
            self.global_step.assign_add(1)
            actor_loss, critic_loss, entropy = self.train(s, visual_s, a, dc_r, old_prob, advantage)
            tf.summary.experimental.set_step(self.global_step)
            if entropy is not None:
                tf.summary.scalar('LOSS/entropy', entropy)
            tf.summary.scalar('LOSS/actor_loss', actor_loss)
            tf.summary.scalar('LOSS/critic_loss', critic_loss)
            tf.summary.scalar('LEARNING_RATE/lr', self.lr)
            tf.summary.scalar('REWARD/discounted_reward', self.data.discounted_reward.values[0].mean())
            tf.summary.scalar('REWARD/reward', self.data.total_reward.values[0].mean())
            self.recorder.writer.flush()
        self.clear()

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, dc_r, old_prob, advantage):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                if self.action_type == 'continuous':
                    mu, sigma, value = self.net(s, visual_s)
                    norm_dist = tfp.distributions.Normal(loc=mu, scale=sigma + self.sigma_offset)
                    new_prob = tf.reduce_mean(norm_dist.prob(a), axis=1, keepdims=True)
                    sample_op = tf.clip_by_value(norm_dist.sample(), -1, 1)
                    entropy = tf.reduce_mean(norm_dist.entropy())
                else:
                    action_probs, value = self.net(s, visual_s)
                    new_prob = tf.reduce_max(action_probs, axis=1, keepdims=True)
                    sample_op = tf.argmax(action_probs, axis=1)
                ratio = new_prob / old_prob
                surrogate = ratio * advantage
                td_error = dc_r - value
                actor_loss = tf.reduce_mean(
                    tf.minimum(
                        surrogate,
                        tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
                    ))
                value_loss = tf.reduce_mean(tf.square(td_error))
                if self.action_type == 'continuous':
                    loss = -(actor_loss - 1.0 * value_loss + self.beta * entropy)
                else:
                    loss = value_loss - actor_loss
            loss_grads = tape.gradient(loss, self.net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(loss_grads, self.net.trainable_variables)
            )
            return actor_loss, value_loss, entropy if self.action_type == 'continuous' else None
