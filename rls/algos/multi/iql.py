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


class IQL(MultiAgentOffPolicy):

    def __init__(self,
                 envspecs,

                 lr=5.0e-4,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 assign_interval=2,
                 share_params=True,
                 network_settings: List[int] = [32, 32],
                 **kwargs):
        assert not any([envspec.is_continuous for envspec in envspecs]), 'IQL only support discrete action space'
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
            value_net_type=OutputNetworkType.CRITIC_QVALUE_ALL,
            value_net_kwargs=dict(output_shape=self.envspecs[i].a_dim, network_settings=network_settings)
        )

        self.q_nets = [_create_net(name='q_net', i=i) for i in range(self.n_models_percopy)]
        self.q_target_nets = [_create_net(name='q_target_net', i=i) for i in range(self.n_models_percopy)]
        self._target_params_update()

        self.lrs = [self.init_lr(lr) for i in range(self.n_models_percopy)]
        self.optimizers = [self.init_optimizer(self.lrs[i]) for i in range(self.n_models_percopy)]

        [self._worker_params_dict.update(self.q_nets[i]._policy_models) for i in range(self.n_models_percopy)]
        [self._all_params_dict.update(self.q_nets[i]._all_models) for i in range(self.n_models_percopy)]
        self._all_params_dict.update({f"optimizer_{i}": optimizer for optimizer in self.optimizers})
        self._model_post_process()
        self.initialize_data_buffer()

    def choose_action(self, obs, evaluation=False):
        actions = []
        for i in range(self.n_agents_percopy):
            j = 0 if self.share_params else i
            if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
                actions.append(np.random.randint(0, self.envspecs[i].a_dim, self.n_copys))
            else:
                a = self._get_action(obs[i], self.q_nets[j])
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
                update_target_net_weights(self.q_target_nets[i].weights, self.q_nets[i].weights)

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn()

    @tf.function
    def _train(self, BATCHs):
        summaries = {}
        with tf.device(self.device):
            for i in range(self.n_agents_percopy):
                j = 0 if self.share_params else i
                with tf.GradientTape() as tape:
                    q = self.q_nets[j](BATCHs[i].obs)['value']
                    q_next = self.q_target_nets[j](BATCHs[i].obs_)['value']
                    q_eval = tf.reduce_sum(tf.multiply(q, BATCHs[i].action), axis=1, keepdims=True)
                    q_target = tf.stop_gradient(BATCHs[i].reward + self.gamma * (1 - BATCHs[i].done) * tf.reduce_max(q_next, axis=1, keepdims=True))
                    td_error = q_target - q_eval
                    q_loss = tf.reduce_mean(tf.square(td_error))
                grads = tape.gradient(q_loss, self.q_nets[j].trainable_variables)
                self.optimizer[j].apply_gradients(
                    zip(grads, self.q_nets[j].trainable_variables)
                )
                # TODO:
                summaries[i] = dict([
                    ['LOSS/loss', q_loss],
                    ['Statistics/q_max', tf.reduce_max(q_eval)],
                    ['Statistics/q_min', tf.reduce_min(q_eval)],
                    ['Statistics/q_mean', tf.reduce_mean(q_eval)]
                ])
            self.global_step.assign_add(1)
            return summaries
