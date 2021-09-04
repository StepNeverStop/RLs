#!/usr/bin/env python3
# encoding: utf-8

from typing import Dict, List, NoReturn, Union

import numpy as np
import torch as t
from torch import distributions as td

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.common.decorator import iton
from rls.common.specs import Data
from rls.nn.dreamer import DenseModel, RecurrentStateSpaceModel
from rls.nn.utils import OPLR


class PlaNet(SarlOffPolicy):
    '''
    Learning Latent Dynamics for Planning from Pixels, http://arxiv.org/abs/1811.04551
    '''
    policy_mode = 'off-policy'

    def __init__(self,

                 stoch_dim=30,
                 deter_dim=200,
                 model_lr=6e-4,
                 kl_free_nats=3,
                 kl_scale=1.0,
                 reward_scale=1.0,
                 cem_horizon=12,
                 cem_iter_nums=10,
                 cem_candidates=1000,
                 cem_tops=100,
                 action_sigma=0.3,
                 network_settings=dict(),
                 **kwargs):
        super().__init__(**kwargs)

        assert self.is_continuous == True, 'assert self.is_continuous == True'

        self.cem_horizon = cem_horizon
        self.cem_iter_nums = cem_iter_nums
        self.cem_candidates = cem_candidates
        self.cem_tops = cem_tops

        assert self.use_rnn == False, 'assert self.use_rnn == False'

        if self.obs_spec.has_visual_observation \
            and len(self.obs_spec.visual_dims) == 1 \
                and not self.obs_spec.has_vector_observation:
            visual_dim = self.obs_spec.visual_dims[0]
            # TODO: optimize this
            assert visual_dim[0] == visual_dim[1] == 64, 'visual dimension must be [64, 64, *]'
            self._is_visual = True
        elif self.obs_spec.has_vector_observation \
            and len(self.obs_spec.vector_dims) == 1 \
                and not self.obs_spec.has_visual_observation:
            self._is_visual = False
        else:
            raise ValueError("please check the observation type")

        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.kl_free_nats = kl_free_nats
        self.kl_scale = kl_scale
        self.reward_scale = reward_scale
        self._action_sigma = action_sigma
        self._network_settings = network_settings

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

        self.model_oplr = OPLR([self.obs_encoder, self.rssm, self.obs_decoder, self.reward_predictor],
                               model_lr, **self._oplr_params)
        self._trainer_modules.update(obs_encoder=self.obs_encoder,
                                     obs_decoder=self.obs_decoder,
                                     reward_predictor=self.reward_predictor,
                                     rssm=self.rssm,
                                     model_oplr=self.model_oplr)

    @property
    def decoder_input_dim(self):
        return self.stoch_dim + self.deter_dim

    def _dreamer_build_rssm(self):
        return RecurrentStateSpaceModel(self.stoch_dim,
                                        self.deter_dim,
                                        self.a_dim,
                                        self.obs_encoder.h_dim,
                                        **self._network_settings['rssm']).to(self.device)

    @iton
    def select_action(self, obs):
        if self._is_visual:
            obs = obs.visual.visual_0
        else:
            obs = obs.vector.vector_0
        # Compute starting state for planning
        # while taking information from current observation (posterior)
        embedded_obs = self.obs_encoder(obs)    # [B, *]
        state_posterior = self.rssm.posterior(self.rnncs['hx'], embedded_obs)     # dist # [B, *]

        # Initialize action distribution
        mean = t.zeros((self.cem_horizon, 1, self.n_copys, self.a_dim))    # [H, 1, B, A]
        stddev = t.ones((self.cem_horizon, 1, self.n_copys, self.a_dim))   # [H, 1, B, A]

        # Iteratively improve action distribution with CEM
        for itr in range(self.cem_iter_nums):
            action_candidates = mean + stddev * t.randn(self.cem_horizon, self.cem_candidates, self.n_copys, self.a_dim)    # [H, N, B, A]
            action_candidates = action_candidates.reshape(self.cem_horizon, -1, self.a_dim)    # [H, N*B, A]

            # Initialize reward, state, and rnn hidden state
            # These are for parallel exploration
            total_predicted_reward = t.zeros((self.cem_candidates*self.n_copys, 1))    # [N*B, 1]

            state = state_posterior.sample((self.cem_candidates,))   # [N, B, *]
            state = state.view(-1, state.shape[-1])  # [N*B, *]
            rnn_hidden = self.rnncs['hx'].repeat((self.cem_candidates, 1))  # [B, *] => [N*B, *]

            # Compute total predicted reward by open-loop prediction using pri
            for _t in range(self.cem_horizon):
                next_state_prior, rnn_hidden = self.rssm.prior(state, t.tanh(action_candidates[_t]), rnn_hidden)
                state = next_state_prior.sample()   # [N*B, *]
                post_feat = t.cat([state, rnn_hidden], -1)  # [N*B, *]
                total_predicted_reward += self.reward_predictor(post_feat).mean  # [N*B, 1]

            # update action distribution using top-k samples
            total_predicted_reward = total_predicted_reward.view(self.cem_candidates, self.n_copys, 1)    # [N, B, 1]
            _, top_indexes = total_predicted_reward.topk(self.cem_tops, dim=0, largest=True, sorted=False)    # [N', B, 1]
            action_candidates = action_candidates.view(self.cem_horizon, self.cem_candidates, self.n_copys, -1)   # [H, N, B, A]
            top_action_candidates = action_candidates[:, top_indexes, t.arange(self.n_copys).reshape(self.n_copys, 1), t.arange(self.a_dim)]  # [H, N', B, A]
            mean = top_action_candidates.mean(dim=1, keepdim=True)    # [H, 1, B, A]
            stddev = top_action_candidates.std(dim=1, unbiased=False, keepdim=True)  # [H, 1, B, A]

        # Return only first action (replan each state based on new observation)
        actions = t.tanh(mean[0].squeeze(0))    # [B, A]
        actions = self._exploration(actions)
        _, self.rnncs_['hx'] = self.rssm.prior(state_posterior.sample(),
                                               actions,
                                               self.rnncs['hx'])
        return actions, Data(action=actions)

    def _exploration(self, action: t.Tensor) -> t.Tensor:
        """
        :param action: action to take, shape (1,) (if categorical), or (action dim,) (if continuous)
        :return: action of the same shape passed in, augmented with some noise
        """
        sigma = self._action_sigma if self._is_train_mode else 0.
        noise = t.randn(*action.shape) * sigma
        return t.clamp(action + noise, -1, 1)

    @iton
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
            # if the begin of this episode, then reset to 0.
            # No matther whether last episode is beened truncated of not.
            state = state * (1. - BATCH.begin_mask[l])  # [B, S]
            rnn_hidden = rnn_hidden * (1. - BATCH.begin_mask[l])     # [B, D]

            next_state_prior, next_state_posterior, rnn_hidden = self.rssm(state,
                                                                           BATCH.action[l],
                                                                           rnn_hidden,
                                                                           embedded_observations[l])    # a, s_
            state = next_state_posterior.rsample()  # [B, S] posterior of s_
            states.append(state)  # [B, S]
            rnn_hiddens.append(rnn_hidden)   # [B, D]
            kl_loss += self._kl_loss(next_state_prior, next_state_posterior)
        kl_loss /= T  # 1

        # compute reconstructed observations and predicted rewards
        post_feat = t.cat([t.stack(states, 0), t.stack(rnn_hiddens, 0)], -1)  # [T, B, *]

        obs_pred = self.obs_decoder(post_feat)  # [T, B, C, H, W] or [T, B, *]
        reward_pred = self.reward_predictor(post_feat)  # [T, B, 1], s_ => r

        # compute loss for observation and reward
        obs_loss = -t.mean(obs_pred.log_prob(obs_))  # [T, B] => 1
        # [T, B, 1]=>1
        reward_loss = -t.mean(reward_pred.log_prob(BATCH.reward).unsqueeze(-1))

        # add all losses and update model parameters with gradient descent
        model_loss = self.kl_scale*kl_loss + obs_loss + self.reward_scale * reward_loss   # 1

        self.model_oplr.optimize(model_loss)

        summaries = dict([
            ['LEARNING_RATE/model_lr', self.model_oplr.lr],
            ['LOSS/model_loss', model_loss],
            ['LOSS/kl_loss', kl_loss],
            ['LOSS/obs_loss', obs_loss],
            ['LOSS/reward_loss', reward_loss]
        ])

        return t.ones_like(BATCH.reward), summaries

    def _initial_rnncs(self, batch: int) -> Dict[str, np.ndarray]:
        return {'hx': np.zeros((batch, self.deter_dim))}

    def _kl_loss(self, prior_dist, post_dist):
        # 1
        return td.kl_divergence(prior_dist, post_dist).clamp(min=self.kl_free_nats).mean()
