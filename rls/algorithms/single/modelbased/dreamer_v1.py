#!/usr/bin/env python3
# encoding: utf-8

from typing import Dict, List, NoReturn, Union

import numpy as np
import torch as t
from torch import distributions as td

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.common.decorator import iTensor_oNumpy
from rls.common.specs import Data
from rls.nn.dreamer import (ActionModel, RecurrentStateSpaceModel, RewardModel,
                            ValueModel)
from rls.nn.utils import OPLR
from rls.utils.expl_expt import ExplorationExploitationClass


def lambda_return(rewards, values, gamma, lambda_):
    V_lambda = t.zeros_like(rewards, device=rewards.device)

    H = rewards.shape[0] - 1
    V_n = t.zeros_like(rewards, device=rewards.device)
    V_n[H] = values[H]
    for n in range(1, H+1):
        V_n[:-n] = (gamma ** n) * values[n:]
        for k in range(1, n+1):
            if k == n:
                V_n[:-n] += (gamma ** (n-1)) * rewards[k:]
            else:
                V_n[:-n] += (gamma ** (k-1)) * rewards[k:-n+k]

        if n == H:
            V_lambda += (lambda_ ** (H-1)) * V_n
        else:
            V_lambda += (1 - lambda_) * (lambda_ ** (n-1)) * V_n

    return V_lambda


class DreamerV1(SarlOffPolicy):
    '''
    Dream to Control: Learning Behaviors by Latent Imagination, http://arxiv.org/abs/1912.01603
    '''
    policy_mode = 'off-policy'

    def __init__(self,

                 eps_init: float = 1,
                 eps_mid: float = 0.2,
                 eps_final: float = 0.01,
                 init2mid_annealing_step: int = 1000,
                 stoch_dim=30,
                 deter_dim=200,
                 model_lr=6e-4,
                 actor_lr=8e-5,
                 critic_lr=8e-5,
                 free_nats=3,
                 imagination_horizon=15,
                 lambda_=0.95,
                 cnn_depth=32,
                 cnn_act="relu",
                 dense_act='elu',
                 vec_feat_dim=16,
                 num_units=400,
                 init_stddev=5.0,
                 **kwargs):
        super().__init__(**kwargs)

        assert self.use_rnn == False, 'assert self.use_rnn == False'

        if self.obs_spec.has_visual_observation and len(
                self.obs_spec.visual_dims) == 1 and not self.obs_spec.has_vector_observation:
            visual_dim = self.obs_spec.visual_dims[0]
            # TODO: optimize this
            assert visual_dim[0] == visual_dim[1] == 64, 'visual dimension must be [64, 64, *]'
            self._is_visual = True
        elif self.obs_spec.has_vector_observation and len(
                self.obs_spec.vector_dims) == 1 and not self.obs_spec.has_visual_observation:
            self._is_visual = False
        else:
            raise ValueError("please check the observation type")

        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.free_nats = free_nats
        self.imagination_horizon = imagination_horizon
        self.lambda_ = lambda_

        if not self.is_continuous:
            self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                              eps_mid=eps_mid,
                                                              eps_final=eps_final,
                                                              init2mid_annealing_step=init2mid_annealing_step,
                                                              max_step=self.max_train_step)

        if self.obs_spec.has_visual_observation:
            from rls.nn.dreamer import VisualDecoder, VisualEncoder
            self.obs_encoder = VisualEncoder(self.obs_spec.visual_dims[0],
                                             depth=cnn_depth,
                                             act=cnn_act).to(self.device)
            self.obs_decoder = VisualDecoder(stoch_dim,
                                             deter_dim,
                                             self.obs_spec.visual_dims[0],
                                             depth=cnn_depth,
                                             act=cnn_act).to(self.device)
        else:
            self._is_visual = False
            from rls.nn.dreamer import VectorDecoder, VectorEncoder
            self.obs_encoder = VectorEncoder(self.obs_spec.vector_dims[0],
                                             vec_feat_dim).to(self.device)
            self.obs_decoder = VectorDecoder(stoch_dim,
                                             deter_dim,
                                             self.obs_spec.vector_dims[0]).to(self.device)

        self.rssm = RecurrentStateSpaceModel(stoch_dim,
                                             self.a_dim,
                                             deter_dim,
                                             self.obs_encoder.h_dim).to(self.device)
        self.reward_predictor = RewardModel(stoch_dim,
                                            deter_dim,
                                            hidden_dim=num_units,
                                            act=dense_act).to(self.device)
        self.actor = ActionModel(stoch_dim,
                                 deter_dim,
                                 self.a_dim,
                                 is_continuous=self.is_continuous,
                                 hidden_dim=num_units,
                                 init_stddev=init_stddev,
                                 act=dense_act).to(self.device)
        self.critic = ValueModel(stoch_dim,
                                 deter_dim,
                                 hidden_dim=num_units,
                                 act=dense_act).to(self.device)

        self.model_oplr = OPLR([self.obs_encoder, self.rssm, self.obs_decoder, self.reward_predictor],
                               model_lr, optimizer_params=dict(eps=1e-4), clipnorm=100)
        self.actor_oplr = OPLR(self.actor, actor_lr,
                               optimizer_params=dict(eps=1e-4), clipnorm=100)
        self.critic_oplr = OPLR(
            self.critic, critic_lr, optimizer_params=dict(eps=1e-4), clipnorm=100)
        self._trainer_modules.update(obs_encoder=self.obs_encoder,
                                     rssm=self.rssm,
                                     obs_decoder=self.obs_decoder,
                                     reward_predictor=self.reward_predictor,
                                     actor=self.actor,
                                     critic=self.critic,
                                     model_oplr=self.model_oplr,
                                     actor_oplr=self.actor_oplr,
                                     critic_oplr=self.critic_oplr)

    @iTensor_oNumpy
    def select_action(self, obs):
        if self._is_visual:
            obs = obs.visual.visual_0
        else:
            obs = obs.vector.vector_0
        embedded_obs = self.obs_encoder(obs)    # [B, *]
        state_posterior = self.rssm.posterior(
            self.cell_state['hx'], embedded_obs)
        state = state_posterior.sample()    # [B, *]
        output = self.actor(state, self.cell_state['hx'])
        if self.is_continuous:
            mu, sigma = output  # [B, A], [B, A]
            if self._is_train_mode:
                actions = t.tanh(td.Normal(mu, sigma).rsample())      # [B, A]
            else:
                actions = t.tanh(mu)    # [B, A]
            prior_actions = actions
        else:
            if self._is_train_mode and self.expl_expt_mng.is_random(self.cur_train_step):
                actions = t.randint(0, self.a_dim, (self.n_copys,))
            else:
                logits = output
                actions = logits.argmax(-1)  # [B, ]
            prior_actions = t.nn.functional.one_hot(
                actions, self.a_dim).float()
        _, self.next_cell_state['hx'] = self.rssm.prior(state,
                                                        prior_actions,
                                                        self.cell_state['hx'])
        return actions, Data(action=actions)

    @iTensor_oNumpy
    def _train(self, BATCH):
        T, B = BATCH.action.shape[:2]
        if self._is_visual:
            obs_ = BATCH.obs_.visual.visual_0
        else:
            obs_ = BATCH.obs_.vector.vector_0

        # embed observations with CNN
        embedded_observations = self.obs_encoder(obs_)  # [T, B, *]

        # prepare Tensor to maintain states sequence and rnn hidden states sequence
        states = t.zeros(T, B, self.stoch_dim, device=self.device)  # [T, B, S]
        rnn_hiddens = t.zeros(T, B, self.deter_dim,
                              device=self.device)  # [T, B, D]

        # initialize state and rnn hidden state with 0 vector
        state = t.zeros(B, self.stoch_dim, device=self.device)  # [B, S]
        rnn_hidden = t.zeros(B, self.deter_dim, device=self.device)  # [B, D]

        # compute state and rnn hidden sequences and kl loss
        kl_loss = 0
        for l in range(T):
            state = state * BATCH.begin_mask[l]
            rnn_hidden = rnn_hidden * BATCH.begin_mask[l]
            next_state_prior, next_state_posterior, rnn_hidden = \
                self.rssm(state, BATCH.action[l], rnn_hidden,
                          embedded_observations[l])    # a, s_
            state = next_state_posterior.rsample()  # [B, S] posterior of s_
            states[l] = state  # [B, S]
            rnn_hiddens[l] = rnn_hidden   # [B, D]
            kl = td.kl.kl_divergence(
                next_state_prior, next_state_posterior).sum(dim=-1)  # [B,]
            kl_loss += kl.clamp(min=self.free_nats).mean()  # 1
        kl_loss /= T  # 1

        # compute reconstructed observations and predicted rewards
        flatten_states = states.view(-1, self.stoch_dim)  # [T*B, S]
        # [T*B, D]
        flatten_rnn_hiddens = rnn_hiddens.view(-1, self.deter_dim)
        recon_observations = self.obs_decoder(
            flatten_states, flatten_rnn_hiddens)
        recon_observations = recon_observations.view(
            T, B, *recon_observations.shape[1:])   # [T, B, C, H, W] or [T, B, *]
        predicted_rewards = self.reward_predictor(flatten_states, flatten_rnn_hiddens).view(
            T, B, -1)   # [T, B, 1]

        # compute loss for observation and reward
        obs_loss = 0.5 * t.nn.functional.mse_loss(
            recon_observations, obs_, reduction='none').mean([0, 1]).sum()   # 1
        reward_loss = 0.5 * \
            t.nn.functional.mse_loss(predicted_rewards, BATCH.reward)  # 1

        # add all losses and update model parameters with gradient descent
        model_loss = kl_loss + obs_loss + reward_loss   # 1
        self.model_oplr.optimize(model_loss)

        # compute target values
        flatten_states = flatten_states.detach()    # [T*B, S]
        flatten_rnn_hiddens = flatten_rnn_hiddens.detach()  # [T*B, D]
        imaginated_states = t.zeros(self.imagination_horizon + 1,   # [H+1, T*B, S]
                                    *flatten_states.shape,
                                    device=flatten_states.device)
        imaginated_rnn_hiddens = t.zeros(self.imagination_horizon + 1,   # [H+1, T*B, D]
                                         *flatten_rnn_hiddens.shape,
                                         device=flatten_rnn_hiddens.device)
        imaginated_states[0] = flatten_states
        imaginated_rnn_hiddens[0] = flatten_rnn_hiddens

        for h in range(1, self.imagination_horizon + 1):
            output = self.actor(
                flatten_states, flatten_rnn_hiddens)   # [T*B, A]
            if self.is_continuous:
                mu, sigma = output
                actions = t.tanh(td.Normal(mu, sigma).rsample())      # [B, A]
            else:
                logits = output
                prob = logits.softmax(-1)   # [B, A]
                one_hot = t.nn.functional.one_hot(
                    logits.argmax(-1), self.a_dim).float()
                actions = one_hot + prob - prob.detach()
            flatten_states_prior, flatten_rnn_hiddens = self.rssm.prior(flatten_states,
                                                                        actions,
                                                                        flatten_rnn_hiddens)
            flatten_states = flatten_states_prior.rsample()  # [T*B, S]
            imaginated_states[h] = flatten_states   # [T*B, S]
            imaginated_rnn_hiddens[h] = flatten_rnn_hiddens  # [T*B, D]

        # [(H+1)*T*B, S]
        flatten_imaginated_states = imaginated_states.view(-1, self.stoch_dim)
        # [(H+1)*T*B, D]
        flatten_imaginated_rnn_hiddens = imaginated_rnn_hiddens.view(
            -1, self.deter_dim)
        imaginated_rewards = \
            self.reward_predictor(flatten_imaginated_states,
                                  flatten_imaginated_rnn_hiddens).view(self.imagination_horizon + 1, -1)    # [(H+1), T*B]
        imaginated_values = \
            self.critic(flatten_imaginated_states,
                        flatten_imaginated_rnn_hiddens).view(self.imagination_horizon + 1, -1)  # [(H+1), T*B]
        lambda_value_target = lambda_return(imaginated_rewards, imaginated_values,
                                            self.gamma, self.lambda_)   # [(H+1), T*B]

        # NOTE: gradient passing problem, fixed by
        # https://discuss.pytorch.org/t/solved-pytorch1-5-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/90256/16

        # update_value model
        critic_loss = 0.5 * \
            t.nn.functional.mse_loss(
                imaginated_values, lambda_value_target.detach())    # 1
        self.critic_oplr.zero_grad()
        self.critic_oplr.backward(
            critic_loss, backward_params=dict(retain_graph=True))

        # update value model and action model
        actor_loss = -1 * (lambda_value_target.mean())
        self.actor_oplr.zero_grad()
        self.actor_oplr.backward(actor_loss)

        self.critic_oplr.step()
        self.actor_oplr.step()

        td_error = (imaginated_values -
                    lambda_value_target).mean(0).detach()  # [T*B,]
        td_error = td_error.view(T, B, 1)

        return td_error, dict([
            ['LEARNING_RATE/model_lr', self.model_oplr.lr],
            ['LEARNING_RATE/actor_lr', self.actor_oplr.lr],
            ['LEARNING_RATE/critic_lr', self.critic_oplr.lr],
            ['LOSS/model_loss', model_loss],
            ['LOSS/kl_loss', kl_loss],
            ['LOSS/obs_loss', obs_loss],
            ['LOSS/reward_loss', reward_loss],
            ['LOSS/actor_loss', actor_loss],
            ['LOSS/critic_loss', critic_loss]
        ])

    def _initial_cell_state(self, batch: int) -> Dict[str, np.ndarray]:
        return {'hx': np.zeros((batch, self.deter_dim))}
