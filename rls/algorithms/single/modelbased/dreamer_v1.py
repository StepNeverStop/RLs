#!/usr/bin/env python3
# encoding: utf-8

from typing import Dict, List, NoReturn, Union

import numpy as np
import torch as t
from torch import distributions as td

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.common.decorator import iTensor_oNumpy
from rls.common.specs import Data
from rls.nn.dreamer import ActionDecoder, DenseModel, RecurrentStateSpaceModel
from rls.nn.dreamer.utils import FreezeParameters
from rls.nn.utils import OPLR
from rls.utils.expl_expt import ExplorationExploitationClass


def compute_return(reward: t.Tensor,
                   value: t.Tensor,
                   discount: t.Tensor,
                   bootstrap: t.Tensor,
                   lambda_: float):
    """
    Compute the discounted reward for a batch of data.
    reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
    Bootstrap is [batch, 1]
    """
    next_values = t.cat([value[1:], bootstrap[None]], 0)
    target = reward + discount * next_values * (1 - lambda_)
    timesteps = list(range(reward.shape[0] - 1, -1, -1))
    outputs = []
    accumulated_reward = bootstrap
    for _t in timesteps:
        inp = target[_t]
        discount_factor = discount[_t]
        accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
        outputs.append(accumulated_reward)
    returns = t.flip(t.stack(outputs), [0])
    return returns


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
                 vec_feat_dim=16,
                 kl_scale=1.0,
                 use_pcont=False,
                 pcont_scale=10.0,
                 network_settings={
                     'actor': {
                         'layers': 3,
                         'hidden_units': 200
                     },
                     'critic': {
                         'layers': 3,
                         'hidden_units': 200
                     },
                     'reward': {
                         'layers': 3,
                         'hidden_units': 300
                     },
                     'pcont': {
                         'layers': 3,
                         'hidden_units': 200
                     }
                 },
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
        self.kl_scale = kl_scale
        # https://github.com/danijar/dreamer/issues/2
        self.use_pcont = use_pcont  # probability of continuing
        self.pcont_scale = pcont_scale

        feat_dim = stoch_dim + deter_dim

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
            self.obs_decoder = VisualDecoder(feat_dim,
                                             self.obs_spec.visual_dims[0],
                                             depth=cnn_depth,
                                             act=cnn_act).to(self.device)
        else:
            self._is_visual = False
            from rls.nn.dreamer import VectorDecoder, VectorEncoder
            self.obs_encoder = VectorEncoder(self.obs_spec.vector_dims[0],
                                             vec_feat_dim).to(self.device)
            self.obs_decoder = VectorDecoder(feat_dim,
                                             self.obs_spec.vector_dims[0]).to(self.device)

        self.rssm = RecurrentStateSpaceModel(stoch_dim,
                                             deter_dim,
                                             self.a_dim,
                                             self.obs_encoder.h_dim).to(self.device)

        """
        p(r_t | s_t, h_t)
        Reward model to predict reward from state and rnn hidden state
        """
        self.reward_predictor = DenseModel(feat_dim,
                                           (1,),
                                           network_settings['reward']['layers'],
                                           network_settings['reward']['hidden_units']).to(self.device)
        if self.is_continuous:
            action_dist = 'tanh_normal'
        else:
            action_dist = 'one_hot'  # 'relaxed_one_hot'
        self.actor = ActionDecoder(self.a_dim,
                                   feat_dim,
                                   network_settings['actor']['layers'],
                                   network_settings['actor']['hidden_units'],
                                   action_dist).to(self.device)
        self.critic = DenseModel(feat_dim,
                                 (1,),
                                 network_settings['critic']['layers'],
                                 network_settings['critic']['hidden_units']).to(self.device)

        _modules = [self.obs_encoder, self.rssm,
                    self.obs_decoder, self.reward_predictor]
        if self.use_pcont:
            self.pcont_decoder = DenseModel(feat_dim,
                                            (1,),
                                            network_settings['pcont']['layers'],
                                            network_settings['pcont']['hidden_units'],
                                            dist='binary')
            _modules.append(self.pcont_decoder)

        self.model_oplr = OPLR(
            _modules, model_lr, optimizer_params=dict(eps=1e-4), clipnorm=100)
        self.actor_oplr = OPLR(self.actor, actor_lr,
                               optimizer_params=dict(eps=1e-4), clipnorm=100)
        self.critic_oplr = OPLR(
            self.critic, critic_lr, optimizer_params=dict(eps=1e-4), clipnorm=100)
        self._trainer_modules.update(obs_encoder=self.obs_encoder,
                                     obs_decoder=self.obs_decoder,
                                     reward_predictor=self.reward_predictor,
                                     rssm=self.rssm,
                                     actor=self.actor,
                                     critic=self.critic,
                                     model_oplr=self.model_oplr,
                                     actor_oplr=self.actor_oplr,
                                     critic_oplr=self.critic_oplr)
        if self.use_pcont:
            self._trainer_modules.update(pcont_decoder=self.pcont_decoder)

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
        actions = self.actor(
            t.cat((state, self.cell_state['hx']), -1), is_train=self._is_train_mode)
        actions = self._exploration(actions)
        _, self.next_cell_state['hx'] = self.rssm.prior(state,
                                                        actions,
                                                        self.cell_state['hx'])
        if not self.is_continuous:
            actions = actions.argmax(-1)    # [B,]
        return actions, Data(action=actions)

    def _exploration(self, action: t.Tensor) -> t.Tensor:
        """
        :param action: action to take, shape (1,) (if categorical), or (action dim,) (if continuous)
        :return: action of the same shape passed in, augmented with some noise
        """
        if self.is_continuous:
            sigma = 0.4 if self._is_train_mode else 0.
            noise = t.randn(*action.shape) * sigma
            return t.clamp(action + noise, -1, 1)
        else:
            if self._is_train_mode and self.expl_expt_mng.is_random(self.cur_train_step):
                action = t.randint(0, self.a_dim, (self.n_copys, ))
                action = t.zeros_like(action)
                action[..., index] = 1
            return action

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
        states = t.zeros(T, B, self.stoch_dim)  # [T, B, S]
        rnn_hiddens = t.zeros(T, B, self.deter_dim)  # [T, B, D]

        # initialize state and rnn hidden state with 0 vector
        state = t.zeros(B, self.stoch_dim)  # [B, S]
        rnn_hidden = t.zeros(B, self.deter_dim)  # [B, D]

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
        post_feat = t.cat([states, rnn_hiddens], -1)  # [T, B, *]
        obs_pred = self.obs_decoder(post_feat)  # [T, B, C, H, W] or [T, B, *]
        reward_pred = self.reward_predictor(post_feat)  # [T, B, 1]

        # compute loss for observation and reward
        obs_loss = -t.mean(obs_pred.log_prob(obs_))
        reward_loss = -t.mean(reward_pred.log_prob(BATCH.reward))   # 1

        # add all losses and update model parameters with gradient descent
        model_loss = self.kl_scale*kl_loss + obs_loss + reward_loss   # 1

        if self.use_pcont:
            pcont_pred = self.pcont_decoder(post_feat)  # [T, B, 1]
            # https://github.com/danijar/dreamer/issues/2#issuecomment-605392659
            pcont_target = self.gamma * (1. - BATCH.done)
            pcont_loss = -t.mean(pcont_pred.log_prob(pcont_target))
            model_loss += self.pcont_scale * pcont_loss

        # remove gradients from previously calculated tensors
        with t.no_grad():
            # [T*B, S]
            flatten_states = states.view(-1, self.stoch_dim).detach()
            # [T*B, D]
            flatten_rnn_hiddens = rnn_hiddens.view(-1, self.deter_dim).detach()

        with FreezeParameters(self.model_oplr.parameters):
            # compute target values
            imaginated_states = []
            imaginated_rnn_hiddens = []

            for h in range(self.imagination_horizon):
                flatten_feat = t.cat(
                    [flatten_states, flatten_rnn_hiddens], -1).detach()
                actions = self.actor(flatten_feat)   # [T*B, A]
                flatten_states_prior, flatten_rnn_hiddens = self.rssm.prior(flatten_states,
                                                                            actions,
                                                                            flatten_rnn_hiddens)
                flatten_states = flatten_states_prior.rsample()  # [T*B, S]
                imaginated_states.append(flatten_states)   # [T*B, S]
                imaginated_rnn_hiddens.append(flatten_rnn_hiddens)  # [T*B, D]

            imaginated_states = t.stack(imaginated_states, 0)   # [H, T*B, S]
            imaginated_rnn_hiddens = t.stack(
                imaginated_rnn_hiddens, 0)   # [H, T*B, D]

        imaginated_feat = t.cat(
            [imaginated_states, imaginated_rnn_hiddens], -1)
        with FreezeParameters(self.model_oplr.parameters + self.critic_oplr.parameters):
            imaginated_rewards = self.reward_predictor(
                imaginated_feat).mean    # [H, T*B, 1]
            imaginated_values = self.critic(
                imaginated_feat).mean  # [H, T*B, 1]

        # Compute the exponential discounted sum of rewards
        if self.use_pcont:
            with FreezeParameters(self.pcont_decoder.parameters()):
                discount_arr = self.pcont_decoder(
                    imaginated_feat).mean  # [H, T*B, 1]
        else:
            discount_arr = self.gamma * \
                t.ones_like(imaginated_rewards)  # [H, T*B, 1]
        returns = compute_return(imaginated_rewards[:-1], imaginated_values[:-1], discount_arr[:-1],
                                 bootstrap=imaginated_values[-1], lambda_=self.lambda_)    # [H-1, T*B, 1]
        # Make the top row 1 so the cumulative product starts with discount^0
        discount_arr = t.cat(
            [t.ones_like(discount_arr[:1]), discount_arr[1:]])  # [H, T*B, 1]
        discount = t.cumprod(discount_arr[:-1], 0)   # [H-1, T*B, 1]
        actor_loss = -t.mean(discount * returns)    # 1

        # Don't let gradients pass through to prevent overwriting gradients.
        # Value Loss
        with t.no_grad():
            value_feat = imaginated_feat[:-1].detach()  # [H-1, T*B, 1]
            value_discount = discount.detach()  # [H-1, T*B, 1]
            value_target = returns.detach()  # [H-1, T*B, 1]

        value_pred = self.critic(value_feat)  # [H-1, T*B, 1]
        log_prob = value_pred.log_prob(value_target)    # [H-1, T*B]
        critic_loss = -t.mean(value_discount * log_prob.unsqueeze(-1))  # 1

        self.model_oplr.zero_grad()
        self.actor_oplr.zero_grad()
        self.critic_oplr.zero_grad()

        self.model_oplr.backward(model_loss)
        self.actor_oplr.backward(actor_loss)
        self.critic_oplr.backward(critic_loss)

        self.model_oplr.step()
        self.actor_oplr.step()
        self.critic_oplr.step()

        td_error = (value_pred.mean-value_target).mean(0).detach()  # [T*B,]
        td_error = td_error.view(T, B, 1)

        summaries = dict([
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
        if self.use_pcont:
            summaries.update(dict([['LOSS/pcont_loss', pcont_loss]]))

        return td_error, summaries

    def _initial_cell_state(self, batch: int) -> Dict[str, np.ndarray]:
        return {'hx': np.zeros((batch, self.deter_dim))}
