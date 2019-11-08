import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import Nn
from utils.sth import sth
from .policy import Policy


class PG(Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 action_type,
                 gamma=0.99,
                 max_episode=50000,
                 batch_size=128,
                 base_dir=None,

                 lr=5.0e-4,
                 epsilon=0.2,
                 epoch=5,
                 hidden_units={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32]
                 },
                 logger2file=False,
                 out_graph=False):
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim_or_list=a_dim_or_list,
            action_type=action_type,
            gamma=gamma,
            max_episode=max_episode,
            base_dir=base_dir,
            policy_mode='ON',
            batch_size=batch_size)
        self.epoch = epoch
        self.epsilon = epsilon
        self.TensorSpecs = self.get_TensorSpecs([self.s_dim], self.visual_dim, [self.a_counts], [1])
        if self.action_type == 'continuous':
            self.net = Nn.actor_mu(self.s_dim, self.visual_dim, self.a_counts, 'pg_net', hidden_units['actor_continuous'])
        else:
            self.net = Nn.actor_discrete(self.s_dim, self.visual_dim, self.a_counts, 'pg_net', hidden_units['actor_discrete'])
        self.lr = tf.keras.optimizers.schedules.PolynomialDecay(lr, self.max_episode, 1e-10, power=1.0)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr(self.episode))
        self.log_std = tf.Variable(initial_value=-0.5 * np.ones(self.a_counts, dtype=np.float32), trainable=True) if self.action_type == 'continuous' else []
        self.generate_recorder(
            logger2file=logger2file,
            model=self
        )
        self.recorder.logger.info('''
　　　ｘｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　　　ｘｘ　　ｘｘ　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　　　　　　　
　　　　　ｘｘｘｘｘｘ　　　　　　　ｘ　　　ｘｘｘｘｘ　　　
　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　ｘｘｘ　　　　
　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　　ｘ　　　　　　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　　ｘｘｘｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　　　　　
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
                mu = self.net(vector_input, visual_input)
                sample_op, _ = self.gaussian_clip_reparam_sample(mu, self.log_std)
            else:
                logits = self.net(vector_input, visual_input)
                norm_dist = tfp.distributions.Categorical(logits)
                sample_op = norm_dist.sample()
        return sample_op

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        self.on_store(s, visual_s, a, r, s_, visual_s_, done)

    def calculate_statistics(self):
        self.data['total_reward'] = sth.discounted_sum(self.data.r.values, 1, 0, self.data.done.values)
        a = np.array(sth.discounted_sum(self.data.r.values, self.gamma, 0, self.data.done.values))
        a -= np.mean(a)
        a /= np.std(a)
        self.data['discounted_reward'] = list(a)

    def get_sample_data(self, index):
        i_data = self.data.iloc[index:index+self.batch_size]
        s = np.vstack(i_data.s.values)
        visual_s = np.vstack(i_data.visual_s.values)
        a = np.vstack(i_data.a.values)
        dc_r = np.vstack(i_data.discounted_reward.values).reshape(-1, 1)
        return s, visual_s, a, dc_r

    def learn(self, **kwargs):
        assert self.batch_size <= self.data.shape[0], "batch_size must less than the length of an episode"
        self.episode = kwargs['episode']
        self.recorder.writer.set_as_default()
        self.calculate_statistics()
        for _ in range(self.epoch):
            for index in range(0, self.data.shape[0], self.batch_size):
                s, visual_s, a, dc_r = [tf.convert_to_tensor(i) for i in self.get_sample_data(index)]
                loss, entropy = self.train.get_concrete_function(
                            *self.TensorSpecs)(s, visual_s, a, dc_r)
        tf.summary.experimental.set_step(self.episode)
        tf.summary.scalar('LOSS/entropy', entropy)
        tf.summary.scalar('LOSS/loss', loss)
        tf.summary.scalar('LEARNING_RATE/lr', self.lr(self.episode))
        self.recorder.writer.flush()
        self.clear()

    @tf.function(experimental_relax_shapes=True)
    def train(self, s, visual_s, a, dc_r):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                if self.action_type == 'continuous':
                    mu = self.net(s, visual_s)
                    log_act_prob = self.gaussian_likelihood(mu, a, self.log_std)
                    entropy = self.gaussian_entropy(self.log_std)
                else:
                    logits = self.net(s, visual_s)
                    logp_all = tf.nn.log_softmax(logits)
                    log_act_prob = tf.reduce_sum(tf.multiply(logp_all, a), axis=1, keepdims=True)
                    entropy = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=1, keepdims=True))
                loss = tf.reduce_mean(log_act_prob * dc_r)
            loss_grads = tape.gradient(loss, self.net.trainable_variables + [self.log_std])
            self.optimizer.apply_gradients(
                zip(loss_grads, self.net.trainable_variables + [self.log_std])
            )
            self.global_step.assign_add(1)
            return loss, entropy
