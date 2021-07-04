#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf

from rls.algos.base.ma_off_policy import MultiAgentOffPolicy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.tf2_utils import update_target_net_weights
from rls.utils.build_networks import (ValueNetwork,
                                      DefaultRepresentationNetwork)
from rls.utils.specs import OutputNetworkType


class VDN(MultiAgentOffPolicy):
    '''
    Value-Decomposition Networks For Cooperative Multi-Agent Learning, http://arxiv.org/abs/1706.05296
    TODO: RNN, multi-step
    '''

    def __init__(self,
                 envspecs,

                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=2,
                 share_params=True,
                 network_settings={
                     'share': [128],
                     'v': [128],
                     'adv': [128]
                 },
                 **kwargs):
        assert not any([envspec.is_continuous for envspec in envspecs]), 'VDN only support discrete action space'
        super().__init__(envspecs=envspecs, **kwargs)
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval
        self.share_params = share_params
        self.n_models_percopy = 1 if self.share_params else self.n_agents_percopy

        def _create_net(name, i): return ValueNetwork(
            name=name+f'_{i}',
            representation_net=DefaultRepresentationNetwork(
                name=name+f'_{i}',
                obs_spec=self.envspecs[i].obs_spec,
                representation_net_params=self.representation_net_params),
            value_net_type=OutputNetworkType.CRITIC_DUELING,
            value_net_kwargs=dict(output_shape=self.envspecs[i].a_dim, network_settings=network_settings)
        )

        self.dueling_nets = [_create_net(name='dueling_net', i=i) for i in range(self.n_models_percopy)]
        self.dueling_target_nets = [_create_net(name='dueling_target_net', i=i) for i in range(self.n_models_percopy)]
        self._target_params_update()

        self.lr = self.init_lr(lr)
        self.optimizer = self.init_optimizer(self.lr)

        [self._worker_params_dict.update(self.dueling_nets[i]._policy_models) for i in range(self.n_models_percopy)]
        [self._all_params_dict.update(self.dueling_nets[i]._all_models) for i in range(self.n_models_percopy)]
        self._all_params_dict.update(optimizer=self.optimizer)
        self._model_post_process()
        self.initialize_data_buffer()

    def choose_action(self, obs, evaluation=False):
        actions = []
        for i in range(self.n_agents_percopy):
            j = 0 if self.share_params else i
            if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
                actions.append(np.random.randint(0, self.envspecs[i].a_dim, self.n_copys))
            else:
                a = self._get_action(obs[i], self.dueling_nets[j])
                actions.append(a.numpy())
        return actions

    @tf.function
    def _get_action(self, obs, net):
        with tf.device(self.device):
            q_values = net(obs)['value']
        return tf.argmax(q_values, axis=-1)

    def _target_params_update(self):
        if self.global_step % self.assign_interval == 0:
            for i in range(self.n_models_percopy):
                update_target_net_weights(self.dueling_target_nets[i].weights, self.dueling_nets[i].weights)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn()

    @property
    def _training_variables(self):
        tv = []
        for net in self.dueling_nets:
            tv += net.trainable_variables
        return tv

    @tf.function
    def _train(self, BATCHs):
        summaries = {}
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                q_target_next_max_all = 0
                reward = tf.zeros_like(BATCHs[0].reward)
                done = tf.zeros_like(BATCHs[0].done)
                for i in range(self.n_agents_percopy):
                    j = 0 if self.share_params else i
                    reward += BATCHs[i].reward
                    done += BATCHs[i].done
                    q = self.dueling_nets[j](BATCHs[i].obs)['value']
                    next_q = self.dueling_nets[j](BATCHs[i].obs_)['value']
                    q_target = self.dueling_target_nets[j](BATCHs[i].obs_)['value']

                    q_eval = tf.reduce_sum(tf.multiply(q, BATCHs[i].action), axis=1, keepdims=True)
                    q_eval_all += q_eval
                    next_max_action = tf.argmax(next_q, axis=1, name='next_action_int')
                    next_max_action_one_hot = tf.one_hot(tf.squeeze(next_max_action), self.envspecs[i].a_dim, 1., 0., dtype=tf.float32)
                    next_max_action_one_hot = tf.cast(next_max_action_one_hot, tf.float32)

                    q_target_next_max = tf.reduce_sum(
                        tf.multiply(q_target, next_max_action_one_hot),
                        axis=1, keepdims=True)
                    q_target_next_max_all += q_target_next_max

                q_target_all = tf.stop_gradient(reward + self.gamma * q_target_next_max_all * (1 - done > 0))
                td_error = q_target_all - q_eval_all
                q_loss = tf.reduce_mean(tf.square(td_error))
            grads = tape.gradient(q_loss, self._training_variables)
            self.optimizer.apply_gradients(
                zip(grads, self._training_variables)
            )
            self.global_step.assign_add(1)
            summaries['model'] = dict([
                ['LOSS/loss', q_loss],
                ['Statistics/q_max', tf.reduce_max(q_eval_all)],
                ['Statistics/q_min', tf.reduce_min(q_eval_all)],
                ['Statistics/q_mean', tf.reduce_mean(q_eval_all)]
            ])
            return summaries
