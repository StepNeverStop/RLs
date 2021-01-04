#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf

from typing import (Union,
                    List,
                    NoReturn)

from rls.algos.base.off_policy import Off_Policy
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.tf2_utils import update_target_net_weights
from rls.utils.build_networks import ValueNetwork
from rls.utils.specs import OutputNetworkType


class AveragedDQN(Off_Policy):
    '''
    Averaged-DQN, http://arxiv.org/abs/1611.01929
    '''

    def __init__(self,
                 envspec,

                 target_k: int = 4,
                 lr: float = 5.0e-4,
                 eps_init: float = 1,
                 eps_mid: float = 0.2,
                 eps_final: float = 0.01,
                 init2mid_annealing_step: int = 1000,
                 assign_interval: int = 1000,
                 network_settings: List[int] = [32, 32],
                 **kwargs):
        assert not envspec.is_continuous, 'dqn only support discrete action space'
        super().__init__(envspec=envspec, **kwargs)
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval
        self.target_k = target_k
        assert self.target_k > 0, "assert self.target_k > 0"
        self.target_nets = []
        self.current_target_idx = 0

        def _create_net(name, representation_net=None): return ValueNetwork(
            name=name,
            representation_net=representation_net,
            value_net_type=OutputNetworkType.CRITIC_QVALUE_ALL,
            value_net_kwargs=dict(output_shape=self.a_dim, network_settings=network_settings)
        )
        self.q_net = _create_net('dqn_q_net', self._representation_net)

        for i in range(self.target_k):
            target_q_net = _create_net(
                'dqn_q_target_net' + str(i),
                self._create_representation_net('_representation_target_net' + str(i))
            )
            update_target_net_weights(target_q_net.weights, self.q_net.weights)
            self.target_nets.append(target_q_net)

        self.lr = self.init_lr(lr)
        self.optimizer = self.init_optimizer(self.lr)

        self._worker_params_dict.update(self.q_net._policy_models)

        self._all_params_dict.update(self.q_net._all_models)
        self._all_params_dict.update(optimizer=self.optimizer)
        self._model_post_process()

    def choose_action(self, obs, evaluation: bool = False) -> np.ndarray:
        if np.random.uniform() < self.expl_expt_mng.get_esp(self.train_step, evaluation=evaluation):
            a = np.random.randint(0, self.a_dim, self.n_agents)
        else:
            a, self.cell_state = self._get_action(obs, self.cell_state)
            a = a.numpy()
        return a

    @tf.function
    def _get_action(self, obs, cell_state):
        with tf.device(self.device):
            q_values, cell_state = self.q_net(obs, cell_state=cell_state)
            for i in range(1, self.target_k):
                target_q_values, _ = self.target_nets[i](obs, cell_state=cell_state)
                q_values += target_q_values
        return tf.argmax(q_values, axis=1), cell_state  # 不取平均也可以

    def _target_params_update(self):
        if self.global_step % self.assign_interval == 0:
            update_target_net_weights(self.target_nets[self.current_target_idx].weights, self.q_net.weights)
            self.current_target_idx = (self.current_target_idx + 1) % self.target_k

    def learn(self, **kwargs) -> NoReturn:
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn(function_dict={
                'summary_dict': dict([['LEARNING_RATE/lr', self.lr(self.train_step)]])
            })

    @tf.function
    def _train(self, BATCH, isw, cell_state):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                q, _ = self.q_net(BATCH.obs, cell_state=cell_state)
                q_next, _ = self.target_nets[0](BATCH.obs_, cell_state=cell_state)
                for i in range(1, self.target_k):
                    target_q_values, _ = self.target_nets[i](BATCH.obs, cell_state=cell_state)
                    q_next += target_q_values
                q_next /= self.target_k
                q_eval = tf.reduce_sum(tf.multiply(q, BATCH.action), axis=1, keepdims=True)
                q_target = tf.stop_gradient(BATCH.reward + self.gamma * (1 - BATCH.done) * tf.reduce_max(q_next, axis=1, keepdims=True))
                td_error = q_target - q_eval
                q_loss = tf.reduce_mean(tf.square(td_error) * isw)
            grads = tape.gradient(q_loss, self.q_net.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.q_net.trainable_variables)
            )
            self.global_step.assign_add(1)
            return td_error, dict([
                ['LOSS/loss', q_loss],
                ['Statistics/q_max', tf.reduce_max(q_eval)],
                ['Statistics/q_min', tf.reduce_min(q_eval)],
                ['Statistics/q_mean', tf.reduce_mean(q_eval)]
            ])
