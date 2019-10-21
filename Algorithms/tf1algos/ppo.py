import numpy as np
import tensorflow as tf
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
                 gamma=0.99,
                 max_episode=50000,
                 batch_size=128,
                 base_dir=None,

                 epoch=5,
                 beta=1.0e-3,
                 lr=5.0e-4,
                 lambda_=0.95,
                 epsilon=0.2,
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
        self.beta = beta
        self.epoch = epoch
        self.lambda_ = lambda_
        self.epsilon = epsilon
        with self.graph.as_default():
            self.dc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_reward')
            self.advantage = tf.placeholder(tf.float32, [None, 1], "advantage")
            self.sigma_offset = tf.placeholder(tf.float32, [self.a_counts, ], 'sigma_offset')
            self.old_prob = tf.placeholder(tf.float32, [None, 1], 'old_prob')
            self.lr = tf.train.polynomial_decay(lr, self.episode, self.max_episode, 1e-10, power=1.0)
            if self.action_type == 'continuous':
                self.mu, self.sigma, self.value = Nn.a_c_v_continuous('ppo_net', self.pl_s, self.pl_visual_s, self.a_counts)
                self.norm_dist = tf.distributions.Normal(loc=self.mu, scale=self.sigma + self.sigma_offset)
                self.new_prob = tf.reduce_mean(self.norm_dist.prob(self.pl_a), axis=1)[:, np.newaxis]
                self.sample_op = tf.clip_by_value(self.norm_dist.sample(), -1, 1)
                self.entropy = self.norm_dist.entropy()
                tf.summary.scalar('LOSS/entropy', tf.reduce_mean(self.entropy))
            else:
                self.action_probs, self.value = Nn.a_c_v_discrete('ppo_net', self.pl_s, self.pl_visual_s, self.a_counts)
                self.new_prob = tf.reduce_max(self.action_probs, axis=1)[:, np.newaxis]
                self.sample_op = tf.argmax(self.action_probs, axis=1)

            self.action = tf.identity(self.sample_op, name='action')
            # ratio = tf.exp(self.new_prob - self.old_prob)
            ratio = self.new_prob / self.old_prob
            surrogate = ratio * self.advantage

            self.actor_loss = tf.reduce_mean(
                tf.minimum(
                    surrogate,
                    tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * self.advantage
                ))
            self.value_loss = tf.reduce_mean(tf.squared_difference(self.dc_r, self.value))
            if self.action_type == 'continuous':
                self.loss = -(self.actor_loss - 1.0 * self.value_loss + self.beta * tf.reduce_mean(self.entropy))
            else:
                self.loss = self.value_loss - self.actor_loss

            self.net_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='ppo_net')

            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=self.net_vars, global_step=self.global_step)
            tf.summary.scalar('LOSS/actor_loss', tf.reduce_mean(self.actor_loss))
            tf.summary.scalar('LOSS/critic_loss', tf.reduce_mean(self.value_loss))
            tf.summary.scalar('LEARNING_RATE/lr', tf.reduce_mean(self.lr))
            self.summaries = tf.summary.merge_all()
            self.generate_recorder(
                logger2file=logger2file,
                graph=self.graph if out_graph else None
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
            return self.sess.run(self.action, feed_dict={
                self.pl_visual_s: visual_s,
                self.pl_s: s,
                self.sigma_offset: np.full(self.a_counts, 0.01)
            })
        else:
            if np.random.uniform() < self.epsilon:
                a = np.random.randint(0, self.a_counts, len(s))
            else:
                a = self.sess.run(self.action, feed_dict={
                    self.pl_visual_s: visual_s,
                    self.pl_s: s
                })
            return sth.int2action_index(a, self.a_dim_or_list)

    def choose_inference_action(self, s, visual_s):
        a = self.sess.run(self.action, feed_dict={
            self.pl_visual_s: visual_s,
            self.pl_s: s,
            self.sigma_offset: np.full(self.a_counts, 0.01)
        })
        return a if self.action_type == 'continuous' else sth.int2action_index(a, self.a_dim_or_list)

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray), "store_data need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store_data need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store_data need done type is np.ndarray"

        self.data = self.data.append({
            's': s,
            'visual_s': visual_s,
            'a': a,
            'r': r,
            'done': done,
            'value': np.squeeze(self.sess.run(self.value, feed_dict={
                self.pl_visual_s: visual_s,
                self.pl_s: s,
                self.sigma_offset: np.full(self.a_counts, 0.01)
            })),
            'prob': self.sess.run(self.new_prob, feed_dict={
                self.pl_visual_s: visual_s,
                self.pl_s: s,
                self.pl_a: a if self.action_type == 'continuous' else sth.action_index2one_hot(a, self.a_dim_or_list),
                self.sigma_offset: np.full(self.a_counts, 0.01)
            }) + 1e-10
        }, ignore_index=True)
        self.s_ = s_
        self.visual_s_ = visual_s_

    def calculate_statistics(self):
        self.data['total_reward'] = sth.discounted_sum(self.data.r.values, 1, 0, self.data.done.values)
        init_value = np.squeeze(self.sess.run(self.value, feed_dict={
            self.pl_visual_s: self.visual_s_,
            self.pl_s: self.s_,
            self.sigma_offset: np.full(self.a_counts, 0.01)
        })),
        self.data['discounted_reward'] = sth.discounted_sum(self.data.r.values, self.gamma, init_value, self.data.done.values)
        self.data['td_error'] = sth.discounted_sum_minus(
            self.data.r.values,
            self.gamma,
            init_value,
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

    def get_sample_data(self):
        i_data = self.data.sample(n=self.batch_size) if self.batch_size < self.data.shape[0] else self.data
        s = np.vstack(i_data.s.values)
        visual_s = np.vstack(i_data.visual_s.values)
        a = np.vstack(i_data.a.values)
        dc_r = np.vstack(i_data.discounted_reward.values).reshape(-1, 1)
        old_prob = np.vstack(i_data.prob.values)
        advantage = np.vstack(i_data.advantage.values).reshape(-1, 1)
        return s, visual_s, a, dc_r, old_prob, advantage

    def learn(self, episode):
        self.calculate_statistics()
        for _ in range(self.epoch):
            s, visual_s, a, dc_r, old_prob, advantage = self.get_sample_data()
            summaries, _ = self.sess.run([self.summaries, self.train_op], feed_dict={
                self.pl_visual_s: visual_s,
                self.pl_s: s,
                self.pl_a: a if self.action_type == 'continuous' else sth.action_index2one_hot(a, self.a_dim_or_list),
                self.dc_r: dc_r,
                self.old_prob: old_prob,
                self.advantage: advantage,
                self.episode: episode,
                self.sigma_offset: np.full(self.a_counts, 0.01)
            })
            self.recorder.writer.add_summary(summaries, self.sess.run(self.global_step))
        self.recorder.writer_summary(
            x=episode,
            ys=[{
                'tag': 'REWARD/discounted_reward',
                'value': self.data.discounted_reward.values[0].mean()
            },
                {
                'tag': 'REWARD/reward',
                'value': self.data.total_reward.values[0].mean()
            }])
        self.clear()
