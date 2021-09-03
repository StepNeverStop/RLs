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
from rls.nn.dreamer.utils import FreezeParameters, compute_return
from rls.nn.utils import OPLR
from rls.utils.expl_expt import ExplorationExploitationClass


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
                 kl_free_nats=3,
                 action_sigma=0.3,
                 imagination_horizon=15,
                 lambda_=0.95,
                 kl_scale=1.0,
                 reward_scale=1.0,
                 use_pcont=False,
                 pcont_scale=10.0,
                 network_settings=dict(),
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
        self.kl_free_nats = kl_free_nats
        self.imagination_horizon = imagination_horizon
        self.lambda_ = lambda_
        self.kl_scale = kl_scale
        self.reward_scale = reward_scale
        # https://github.com/danijar/dreamer/issues/2
        self.use_pcont = use_pcont  # probability of continuing
        self.pcont_scale = pcont_scale
        self._action_sigma = action_sigma
        self._network_settings = network_settings

        if not self.is_continuous:
            self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                              eps_mid=eps_mid,
                                                              eps_final=eps_final,
                                                              init2mid_annealing_step=init2mid_annealing_step,
                                                              max_step=self._max_train_step)

        if self.obs_spec.has_visual_observation:
            from rls.nn.dreamer import VisualDecoder, VisualEncoder
            self.obs_encoder = VisualEncoder(self.obs_spec.visual_dims[0],
                                             **network_settings['obs_encoder']['visual']).to(self.device)
            self.obs_decoder = VisualDecoder(self.decoder_input_dim,
                                             self.obs_spec.visual_dims[0],
                                             **network_settings['obs_decoder']['visual']).to(self.device)
        else:
            from rls.nn.dreamer import VectorEncoder
            self.obs_encoder = VectorEncoder(self.obs_spec.vector_dims[0],
                                             **network_settings['obs_encoder']['vector']).to(self.device)
            self.obs_decoder = DenseModel(self.decoder_input_dim,
                                          self.obs_spec.vector_dims[0],
                                          **network_settings['obs_decoder']['vector']).to(self.device)

        self.rssm = self._dreamer_build_rssm()

        """
        p(r_t | s_t, h_t)
        Reward model to predict reward from state and rnn hidden state
        """
        self.reward_predictor = DenseModel(self.decoder_input_dim,
                                           1,
                                           **network_settings['reward']).to(self.device)

        self.actor = ActionDecoder(self.a_dim,
                                   self.decoder_input_dim,
                                   dist=self._action_dist,
                                   **network_settings['actor']).to(self.device)
        self.critic = self._dreamer_build_critic()

        _modules = [self.obs_encoder, self.rssm,
                    self.obs_decoder, self.reward_predictor]
        if self.use_pcont:
            self.pcont_decoder = DenseModel(self.decoder_input_dim,
                                            1,
                                            **network_settings['pcont'])
            _modules.append(self.pcont_decoder)

        self.model_oplr = OPLR(
            _modules, model_lr, optimizer_params=self._optim_params, clipnorm=100)
        self.actor_oplr = OPLR(self.actor, actor_lr,
                               optimizer_params=self._optim_params, clipnorm=100)
        self.critic_oplr = OPLR(
            self.critic, critic_lr, optimizer_params=self._optim_params, clipnorm=100)
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

    @property
    def _action_dist(self):
        return 'tanh_normal' if self.is_continuous else 'one_hot'  # 'relaxed_one_hot'

    @property
    def decoder_input_dim(self):
        return self.stoch_dim + self.deter_dim

    def _dreamer_build_rssm(self):
        return RecurrentStateSpaceModel(self.stoch_dim,
                                        self.deter_dim,
                                        self.a_dim,
                                        self.obs_encoder.h_dim,
                                        **self._network_settings['rssm']).to(self.device)

    def _dreamer_build_critic(self):
        return DenseModel(self.decoder_input_dim,
                          1,
                          **self._network_settings['critic']).to(self.device)

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
        actions = self.actor.sample_actions(
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
            sigma = self._action_sigma if self._is_train_mode else 0.
            noise = t.randn(*action.shape) * sigma
            return t.clamp(action + noise, -1, 1)
        else:
            if self._is_train_mode and self.expl_expt_mng.is_random(self._cur_train_step):
                index = t.randint(0, self.a_dim, (self.n_copys, ))
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

        # initialize state and rnn hidden state with 0 vector
        state, rnn_hidden = self.rssm.init_state(shape=B)   # [B, S], [B, D]

        # compute state and rnn hidden sequences and kl loss
        kl_loss = 0
        states, rnn_hiddens = [], []
        for l in range(T):
            state = state * (1. - BATCH.done[l])
            rnn_hidden = rnn_hidden * (1. - BATCH.done[l])
            pre_action = BATCH.action[l] * (1. - BATCH.done[l])
            if l > 0:
                # begin_mask_{t} and not done_{t-1}, set state to zero
                trunced_mask = t.logical_and(
                    BATCH.begin_mask[l], 1. - BATCH.done[l-1]).float()
                state = state * (1. - trunced_mask)
                rnn_hidden = rnn_hidden * (1. - trunced_mask)
            next_state_prior, next_state_posterior, rnn_hidden = \
                self.rssm(state, pre_action, rnn_hidden,
                          embedded_observations[l])    # a, s_
            state = next_state_posterior.rsample()  # [B, S] posterior of s_
            states.append(state)  # [B, S]
            rnn_hiddens.append(rnn_hidden)   # [B, D]
            kl_loss += self._kl_loss(next_state_prior, next_state_posterior)
        kl_loss /= T  # 1

        # compute reconstructed observations and predicted rewards
        post_feat = t.cat(
            [t.stack(states, 0), t.stack(rnn_hiddens, 0)], -1)  # [T, B, *]

        obs_pred = self.obs_decoder(post_feat)  # [T, B, C, H, W] or [T, B, *]
        reward_pred = self.reward_predictor(
            post_feat[:-1])  # [T-1, B, 1], s_ => r

        # compute loss for observation and reward
        obs_loss = -t.mean(obs_pred.log_prob(obs_))  # [T, B] => 1
        # [T-1, B, 1]=>1
        reward_loss = -t.mean(reward_pred.log_prob(BATCH.reward[1:]))

        # add all losses and update model parameters with gradient descent
        model_loss = self.kl_scale*kl_loss + obs_loss + \
            self.reward_scale * reward_loss   # 1

        if self.use_pcont:
            pcont_pred = self.pcont_decoder(
                post_feat[:-1])  # [T-1, B, 1], s_ => done
            # https://github.com/danijar/dreamer/issues/2#issuecomment-605392659
            pcont_target = self.gamma * (1. - BATCH.done)
            pcont_target = pcont_target[1:]
            # [T-1, B, 1]=>1
            pcont_loss = -t.mean(pcont_pred.log_prob(pcont_target))
            model_loss += self.pcont_scale * pcont_loss

        self.model_oplr.optimize(model_loss)

        # remove gradients from previously calculated tensors
        with t.no_grad():
            # [T, B, S] => [T*B, S]
            flatten_states = t.cat(states, 0).detach()
            # [T, B, D] => [T*B, D]
            flatten_rnn_hiddens = t.cat(rnn_hiddens, 0).detach()

        with FreezeParameters(self.model_oplr.parameters):
            # compute target values
            imaginated_states = []
            imaginated_rnn_hiddens = []
            log_probs = []
            entropies = []

            for h in range(self.imagination_horizon):
                flatten_feat = t.cat(
                    [flatten_states, flatten_rnn_hiddens], -1).detach()
                action_dist = self.actor(flatten_feat)
                actions = action_dist.rsample()  # [T*B, A]
                log_probs.append(action_dist.log_prob(
                    actions.detach()).unsqueeze(-1))   # [T*B, 1]
                entropies.append(
                    action_dist.entropy().unsqueeze(-1))    # [T*B, 1]
                flatten_states_prior, flatten_rnn_hiddens = self.rssm.prior(flatten_states,
                                                                            actions,
                                                                            flatten_rnn_hiddens)
                flatten_states = flatten_states_prior.rsample()  # [T*B, S]
                imaginated_states.append(flatten_states)   # [T*B, S]
                imaginated_rnn_hiddens.append(flatten_rnn_hiddens)  # [T*B, D]

            imaginated_states = t.stack(imaginated_states, 0)   # [H, T*B, S]
            imaginated_rnn_hiddens = t.stack(
                imaginated_rnn_hiddens, 0)   # [H, T*B, D]
            log_probs = t.stack(log_probs, 0)  # [H, T*B, 1]
            entropies = t.stack(entropies, 0)  # [H, T*B, 1]

        imaginated_feats = t.cat(
            [imaginated_states, imaginated_rnn_hiddens], -1)    # [H, T*B, *]

        with FreezeParameters(self.model_oplr.parameters + self.critic_oplr.parameters):
            imaginated_rewards = self.reward_predictor(
                imaginated_feats).mean    # [H, T*B, 1]
            imaginated_values = self._dreamer_target_img_value(
                imaginated_feats)   # [H, T*B, 1]]

        # Compute the exponential discounted sum of rewards
        if self.use_pcont:
            with FreezeParameters(self.pcont_decoder.parameters()):
                discount_arr = self.pcont_decoder(
                    imaginated_feats).mean  # [H, T*B, 1]
        else:
            discount_arr = self.gamma * \
                t.ones_like(imaginated_rewards)  # [H, T*B, 1]
        returns = compute_return(imaginated_rewards[:-1], imaginated_values[:-1], discount_arr[:-1],
                                 bootstrap=imaginated_values[-1], lambda_=self.lambda_)    # [H-1, T*B, 1]
        # Make the top row 1 so the cumulative product starts with discount^0
        discount_arr = t.cat(
            [t.ones_like(discount_arr[:1]), discount_arr[:-1]], 0)  # [H, T*B, 1]
        discount = t.cumprod(discount_arr, 0).detach()[:-1]   # [H-1, T*B, 1]

        # discount_arr = t.cat([t.ones_like(discount_arr[:1]), discount_arr[1:]])
        # discount = t.cumprod(discount_arr[:-1], 0)

        actor_loss = self._dreamer_build_actor_loss(
            imaginated_feats, log_probs, entropies, discount, returns)   # 1

        # Don't let gradients pass through to prevent overwriting gradients.
        # Value Loss
        with t.no_grad():
            value_feat = imaginated_feats[:-1].detach()  # [H-1, T*B, 1]
            value_target = returns.detach()  # [H-1, T*B, 1]

        value_pred = self.critic(value_feat)  # [H-1, T*B, 1]
        log_prob = value_pred.log_prob(
            value_target).unsqueeze(-1)    # [H-1, T*B, 1]
        critic_loss = -t.mean(discount * log_prob)  # 1

        self.actor_oplr.zero_grad()
        self.critic_oplr.zero_grad()

        self.actor_oplr.backward(actor_loss)
        self.critic_oplr.backward(critic_loss)

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

    def _kl_loss(self, prior_dist, post_dist):
        # 1
        return td.kl_divergence(prior_dist, post_dist).clamp(min=self.kl_free_nats).mean()

    def _dreamer_target_img_value(self, imaginated_feats):
        imaginated_values = self.critic(
            imaginated_feats).mean  # [H, T*B, 1]
        return imaginated_values

    def _dreamer_build_actor_loss(self, imaginated_feats, log_probs, entropies, discount, returns):
        actor_loss = -t.mean(discount * returns)    # [H-1, T*B, 1] => 1
        return actor_loss
