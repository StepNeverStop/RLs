import rls
import numpy as np
import tensorflow as tf

from typing import List
from algos.tf2algos.base.ma_off_policy import MultiAgentOffPolicy


class MADDPG(MultiAgentOffPolicy):
    '''
    Multi-Agent Deep Deterministic Policy Gradient, https://arxiv.org/abs/1706.02275
    '''

    def __init__(self,
                 s_dim,
                 a_dim,
                 is_continuous,

                 ployak=0.995,
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 hidden_units={
                     'actor': [32, 32],
                     'q': [32, 32]
                 },
                 **kwargs):
        assert all(is_continuous), 'maddpg only support continuous action space'
        super().__init__(
            s_dim=s_dim,
            a_dim=a_dim,
            is_continuous=is_continuous,
            **kwargs)
        self.ployak = ployak

        # self.action_noises = rls.NormalActionNoise(mu=np.zeros(self.a_dim), sigma=1 * np.ones(self.a_dim))
        self.action_noises = {i: rls.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_dim[i]), sigma=0.2 * np.ones(self.a_dim[i])) for i in range(self.agent_sep_ctls)}

        def _actor_net(i): return rls.actor_dpg(self.s_dim[i], self.a_dim[i], hidden_units['actor'])
        self.actor_nets = {i: _actor_net(i) for i in range(self.agent_sep_ctls)}
        self.actor_target_nets = {i: _actor_net(i) for i in range(self.agent_sep_ctls)}

        def _q_net(): return rls.critic_q_one(self.total_s_dim, self.total_a_dim, hidden_units['q'])
        self.q_nets = {i: _q_net() for i in range(self.agent_sep_ctls)}
        self.q_target_nets = {i: _q_net() for i in range(self.agent_sep_ctls)}

        for i in range(self.agent_sep_ctls):
            self.update_target_net_weights(
                self.actor_target_nets[i].weights + self.q_target_nets[i].weights,
                self.actor_nets[i].weights + self.q_nets[i].weights
            )

        self.actor_lrs = {i: self.init_lr(actor_lr) for i in range(self.agent_sep_ctls)}
        self.critic_lrs = {i: self.init_lr(critic_lr) for i in range(self.agent_sep_ctls)}
        self.optimizer_actors = {i: self.init_optimizer(self.actor_lrs[i]) for i in range(self.agent_sep_ctls)}
        self.optimizer_critics = {i: self.init_optimizer(self.critic_lrs[i]) for i in range(self.agent_sep_ctls)}

        models_and_optimizers = {}
        models_and_optimizers.update({f'actor-{i}': self.actor_nets[i] for i in range(self.agent_sep_ctls)})
        models_and_optimizers.update({f'critic-{i}': self.q_nets[i] for i in range(self.agent_sep_ctls)})
        models_and_optimizers.update({f'optimizer_actor-{i}': self.optimizer_actors[i] for i in range(self.agent_sep_ctls)})
        models_and_optimizers.update({f'optimizer_critic-{i}': self.optimizer_critics[i] for i in range(self.agent_sep_ctls)})

        self.model_recorder(models_and_optimizers)

    def show_logo(self):
        self.recorder.logger.info('''
　　ｘｘｘｘ　　　　ｘｘｘ　　　　　　　　　ｘｘ　　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　ｘｘｘ　　　　ｘｘ　　　　　　　　　ｘｘｘ　　　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　　　ｘｘｘ　　ｘｘｘ　　　　　　　　　ｘｘｘ　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　ｘｘｘ　　ｘｘｘ　　　　　　　　　ｘ　ｘｘ　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　　　　　　　
　　　　ｘｘｘｘ　ｘ　ｘ　　　　　　　　ｘｘ　ｘｘ　　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘ　　　ｘｘｘｘｘ　　　
　　　　ｘ　ｘｘｘｘ　ｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　ｘｘｘ　　　　
　　　　ｘ　ｘｘｘ　　ｘ　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　ｘ　　ｘｘ　　ｘ　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　ｘｘｘｘ　ｘｘｘｘｘｘ　　　　　ｘｘｘ　　ｘｘｘｘｘ　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　
        ''')

    def choose_action(self, s: List[np.ndarray], visual_s: List[np.ndarray], evaluation=False) -> List[np.ndarray]:
        '''
        params:
            s List[np.ndarray]: [agent_sep_ctls, batch, dim]
        return:
            a List[np.ndarray]: [agent_sep_ctls, batch, dim]
        '''
        return [self._get_actions(i, s[i], evaluation) for i in range(self.agent_sep_ctls)]

    def _get_actions(self, model_idx, vector_input, evaluation, use_target=False):

        if use_target:
            actor_net = self.actor_target_nets[model_idx]
        else:
            actor_net = self.actor_nets[model_idx]

        @tf.function
        def _get_action_func(vector_input):
            vector_input = self.cast(vector_input)[0]
            with tf.device(self.device):
                return actor_net(vector_input)

        action = _get_action_func(vector_input).numpy()

        if evaluation:
            np.clip(action + self.action_noises[model_idx](), -1, 1, out=action)

        return action

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            if self.data.is_lg_batch_size:
                self.intermediate_variable_reset()
                batch_data = self.data.sample()
                done = batch_data[-1]
                s, visual_a, a, r, s_, visual_s_ = [batch_data[i:i + self.agent_sep_ctls] for i in range(0, len(batch_data) - 1, self.agent_sep_ctls)]
                target_a = [self._get_actions(i, s_[i], evaluation=True, use_target=True) for i in range(self.agent_sep_ctls)]

                s_all = np.hstack(s)
                a_all = np.hstack(a)
                s_next_all = np.hstack(s_)
                target_a_all = np.hstack(target_a)

                for i in range(self.agent_sep_ctls):
                    summary = {}
                    if i == 0:
                        al = np.full(fill_value=[], shape=(done.shape[0], 0), dtype=np.float32)
                        ar = np.hstack(a[i + 1:])
                    elif i == self.agent_sep_ctls - 1:
                        al = np.hstack(a[:i])
                        ar = np.full(fill_value=[], shape=(done.shape[0], 0), dtype=np.float32)
                    else:
                        al = np.hstack(a[:i])
                        ar = np.hstack(a[i + 1:])

                    # actor: al, ar, s(all), s
                    # critic: r, done, s_(all), target_a(all), s(all), a(all)
                    summary.update(self._train(i, s_all, a_all, s_next_all, target_a_all, r[i], done, s[i], al, ar))
                    summary.update({'LEARNING_RATE/actor_lr': self.actor_lrs[i](self.train_step), 'LEARNING_RATE/critic_lr': self.critic_lrs[i](self.train_step)})
                    self.write_training_summaries(self.global_step, summary, self.writers[i])

                self.global_step.assign_add(1)

                for i in range(self.agent_sep_ctls):
                    self.update_target_net_weights(
                        self.actor_target_nets[i].weights + self.q_target_nets[i].weights,
                        self.actor_nets[i].weights + self.q_nets[i].weights,
                        self.ployak)

    def _train(self, model_idx, s, a, s_, a_, r, done, s_i, al, ar):

        q_net = self.q_nets[model_idx]
        q_target_net = self.q_target_nets[model_idx]
        actor_net = self.actor_nets[model_idx]
        optimizer_actor = self.optimizer_actors[model_idx]
        optimizer_critic = self.optimizer_critics[model_idx]

        s, a, s_, a_, r, done, s_i, al, ar = map(self.data_convert, (s, a, s_, a_, r, done, s_i, al, ar))

        @tf.function(experimental_relax_shapes=True)
        def train_persistent(s, s_, a, a_, r, done, s_i, al, ar):
            with tf.device(self.device):
                with tf.GradientTape(persistent=True) as tape:
                    q = q_net(s, a)
                    q_target = q_target_net(s_, a_)
                    dc_r = tf.stop_gradient(r + self.gamma * q_target * (1 - done))
                    td_error = q - dc_r
                    q_loss = 0.5 * tf.reduce_mean(tf.square(td_error))

                    mu = actor_net(s_i)
                    amua = tf.concat((al, mu, ar), axis=-1)
                    q_actor = q_net(s, amua)
                    actor_loss = -tf.reduce_mean(q_actor)
                optimizer_critic.apply_gradients(
                    zip(tape.gradient(q_loss, q_net.trainable_variables), q_net.trainable_variables)
                )
                optimizer_actor.apply_gradients(
                    zip(tape.gradient(actor_loss, actor_net.trainable_variables), actor_net.trainable_variables)
                )
                return dict([
                    ['LOSS/actor_loss', actor_loss],
                    ['LOSS/critic_loss', q_loss]
                ])

        return train_persistent(s, s_, a, a_, r, done, s_i, al, ar)
