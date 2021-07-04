#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from typing import (List,
                    Union,
                    NoReturn,
                    Dict)

from rls.utils.build_networks import (ValueNetwork,
                                      DefaultRepresentationNetwork,
                                      MultiAgentCentralCriticRepresentationNetwork)
from rls.nn.noise import OrnsteinUhlenbeckNoisedAction
from rls.algos.base.ma_off_policy import MultiAgentOffPolicy
from rls.utils.tf2_utils import update_target_net_weights
from rls.utils.specs import OutputNetworkType


class MADDPG(MultiAgentOffPolicy):
    '''
    Multi-Agent Deep Deterministic Policy Gradient, https://arxiv.org/abs/1706.02275
    '''

    def __init__(self,
                 envspecs,

                 ployak=0.995,
                 actor_lr=5.0e-4,
                 critic_lr=1.0e-3,
                 discrete_tau=1.0,
                 share_params=True,
                 network_settings={
                     'actor_continuous': [32, 32],
                     'actor_discrete': [32, 32],
                     'q': [32, 32]
                 },
                 **kwargs):
        '''
        TODO: Annotation
        '''
        super().__init__(envspecs=envspecs, **kwargs)
        self.ployak = ployak
        self.discrete_tau = discrete_tau
        self.share_params = share_params
        self.n_models_percopy = 1 if self.share_params else self.n_agents_percopy

        def _create_actor_net(name, i):
            return ValueNetwork(
                name=name,
                representation_net=DefaultRepresentationNetwork(
                    name=name+f'_{i}',
                    obs_spec=self.envspecs[i].obs_spec,
                    representation_net_params=self.representation_net_params),
                value_net_type=OutputNetworkType.ACTOR_DPG if self.envspecs[i].is_continuous else OutputNetworkType.ACTOR_DCT,
                value_net_kwargs=dict(output_shape=self.envspecs[i].a_dim,
                                      network_settings=network_settings['actor_continuous'] if self.envspecs[i].is_continuous else network_settings['actor_discrete'])
            )

        def _create_critic_net(name, i):
            return ValueNetwork(
                name=name,
                representation_net=MultiAgentCentralCriticRepresentationNetwork(
                    name=name+f'_{i}',
                    obs_spec_list=[self.envspecs[i].obs_spec for i in range(self.n_agents_percopy)],
                    representation_net_params=self.representation_net_params),
                value_net_type=OutputNetworkType.CRITIC_QVALUE_ONE,
                value_net_kwargs=dict(action_dim=sum([envspec.a_dim for envspec in self.envspecs]),
                                      network_settings=network_settings['q'])
            )

        self.actor_nets = [_create_actor_net(name='actor_net', i=i) for i in range(self.n_models_percopy)]
        self.actor_target_nets = [_create_actor_net(name='actor_target_net', i=i) for i in range(self.n_models_percopy)]
        self.critic_nets = [_create_critic_net(name='critic_net', i=i) for i in range(self.n_models_percopy)]
        self.critic_target_nets = [_create_critic_net(name='critic_target_net', i=i) for i in range(self.n_models_percopy)]

        self._target_params_update()

        self.actor_lrs = [self.init_lr(actor_lr) for i in range(self.n_models_percopy)]
        self.critic_lrs = [self.init_lr(critic_lr) for i in range(self.n_models_percopy)]
        self.optimizer_actors = [self.init_optimizer(self.actor_lrs[i]) for i in range(self.n_models_percopy)]
        self.optimizer_critics = [self.init_optimizer(self.critic_lrs[i]) for i in range(self.n_models_percopy)]

        # TODO: 添加动作类型判断
        self.noised_actions = [OrnsteinUhlenbeckNoisedAction(sigma=0.2) for i in range(self.n_models_percopy)]
        self.gumbel_dists = [tfp.distributions.Gumbel(0, 1) for i in range(self.n_models_percopy)]

        [self._worker_params_dict.update(net._policy_models) for net in self.actor_nets]
        [self._all_params_dict.update(net._all_models) for net in self.actor_nets]
        [self._all_params_dict.update(net._all_models) for net in self.critic_nets]
        self._all_params_dict.update({f'optimizer_actor-{i}': self.optimizer_actors[i] for i in range(self.n_models_percopy)})
        self._all_params_dict.update({f'optimizer_critic-{i}': self.optimizer_critics[i] for i in range(self.n_models_percopy)})

        self._model_post_process()
        self.initialize_data_buffer()

    def reset(self):
        super().reset()
        for noised_action in self.noised_actions:
            noised_action.reset()

    def choose_action(self, obs: List, evaluation=False):
        actions = []
        for i in range(self.n_agents_percopy):
            j = 0 if self.share_params else i
            output = self._get_action(obs[i], self.actor_nets[j])
            if self.envspecs[i].is_continuous:
                mu = output
                pi = self.noised_actions[j](mu)
            else:
                logits = output
                mu = tf.argmax(logits, axis=1)
                cate_dist = tfp.distributions.Categorical(logits=logits)
                pi = cate_dist.sample()
            acts = mu.numpy() if evaluation else pi.numpy()
            actions.append(acts)
        return actions

    @tf.function
    def _get_action(self, obs, net):
        with tf.device(self.device):
            output = net(obs)['value']
            return output

    def _target_params_update(self):
        for i in range(self.n_models_percopy):
            update_target_net_weights(
                self.actor_target_nets[i].weights + self.critic_target_nets[i].weights,
                self.actor_nets[i].weights + self.critic_nets[i].weights
            )

    def learn(self, **kwargs) -> NoReturn:
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            self._learn()

    @tf.function
    def _train(self, BATCHs):
        '''
        TODO: Annotation
        '''
        summaries = {}
        with tf.device(self.device):
            target_actions = []
            for i in range(self.n_agents_percopy):
                j = 0 if self.share_params else i
                if self.envspecs[i].is_continuous:
                    target_actions.append(self.actor_target_nets[j](BATCHs[i].obs_)['value'])
                else:
                    target_logits = self.actor_target_nets[j](BATCHs[i].obs_)['value']
                    target_cate_dist = tfp.distributions.Categorical(logits=target_logits)
                    target_pi = target_cate_dist.sample()
                    action_target = tf.one_hot(target_pi, self.envspecs[i].a_dim, dtype=tf.float32)
                    target_actions.append(action_target)
            target_actions = tf.concat(target_actions, axis=-1)

            q_targets = []
            for i in range(self.n_agents_percopy):
                j = 0 if self.share_params else i
                q_targets.append(self.critic_target_nets[j]([BATCH.obs_ for BATCH in BATCHs], target_actions)['value'])

            for i in range(self.n_agents_percopy):
                j = 0 if self.share_params else i
                with tf.GradientTape(persistent=True) as tape:
                    if self.envspecs[i].is_continuous:
                        mu = self.actor_nets[j](BATCHs[i].obs)['value']
                    else:
                        gumbel_noise = tf.cast(self.gumbel_dists[j].sample(BATCHs[i].action.shape), dtype=tf.float32)
                        logits = self.actor_nets[j](BATCHs[i].obs)['value']
                        logp_all = tf.nn.log_softmax(logits)
                        _pi = tf.nn.softmax((logp_all + gumbel_noise) / self.discrete_tau)
                        _pi_true_one_hot = tf.one_hot(tf.argmax(_pi, axis=-1), self.envspecs[i].a_dim)
                        _pi_diff = tf.stop_gradient(_pi_true_one_hot - _pi)
                        mu = _pi_diff + _pi

                    q_actor = self.critic_nets[j](
                        [BATCH.obs for BATCH in BATCHs],
                        tf.concat([BATCH.action for BATCH in BATCHs[:i]]+[mu]+[BATCH.action for BATCH in BATCHs[i+1:]], axis=-1)
                    )['value']
                    actor_loss = -tf.reduce_mean(q_actor)

                    q = self.critic_nets[j](
                        [BATCH.obs for BATCH in BATCHs],
                        tf.concat([BATCH.action for BATCH in BATCHs], axis=-1)
                    )['value']
                    dc_r = tf.stop_gradient(BATCHs[i].reward + self.gamma * q_targets[i] * (1 - BATCHs[i].done))

                    td_error = dc_r - q
                    q_loss = 0.5 * tf.reduce_mean(tf.square(td_error))

                self.optimizer_critics[j].apply_gradients(
                    zip(tape.gradient(q_loss, self.critic_nets[j].trainable_variables),
                        self.critic_nets[j].trainable_variables)
                )
                self.optimizer_actors[j].apply_gradients(
                    zip(tape.gradient(actor_loss, self.actor_nets[j].trainable_variables),
                        self.actor_nets[j].trainable_variables)
                )
                summaries[i] = dict([
                    ['LOSS/actor_loss', actor_loss],
                    ['LOSS/critic_loss', q_loss],
                    ['Statistics/q_min', tf.reduce_min(q)],
                    ['Statistics/q_mean', tf.reduce_mean(q)],
                    ['Statistics/q_max', tf.reduce_max(q)]
                ])
        self.global_step.assign_add(1)
        return summaries
