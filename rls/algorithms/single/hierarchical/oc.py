#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t
from torch import distributions as td

from rls.algorithms.base.sarl_off_policy import SarlOffPolicy
from rls.common.decorator import iTensor_oNumpy
from rls.common.specs import Data
from rls.nn.models import CriticQvalueAll, OcIntraOption
from rls.nn.modules.wrappers import TargetTwin
from rls.nn.utils import OPLR
from rls.utils.expl_expt import ExplorationExploitationClass
from rls.utils.np_utils import int2one_hot
from rls.utils.torch_utils import n_step_return


class OC(SarlOffPolicy):
    '''
    The Option-Critic Architecture. http://arxiv.org/abs/1609.05140
    '''
    policy_mode = 'off-policy'

    def __init__(self,
                 q_lr=5.0e-3,
                 intra_option_lr=5.0e-4,
                 termination_lr=5.0e-4,
                 use_eps_greedy=False,
                 eps_init=1,
                 eps_mid=0.2,
                 eps_final=0.01,
                 init2mid_annealing_step=1000,
                 boltzmann_temperature=1.0,
                 options_num=4,
                 ent_coff=0.01,
                 double_q=False,
                 use_baseline=True,
                 terminal_mask=True,
                 termination_regularizer=0.01,
                 assign_interval=1000,
                 network_settings={
                     'q': [32, 32],
                     'intra_option': [32, 32],
                     'termination': [32, 32]
                 },
                 **kwargs):
        super().__init__(**kwargs)
        self.expl_expt_mng = ExplorationExploitationClass(eps_init=eps_init,
                                                          eps_mid=eps_mid,
                                                          eps_final=eps_final,
                                                          init2mid_annealing_step=init2mid_annealing_step,
                                                          max_step=self.max_train_step)
        self.assign_interval = assign_interval
        self.options_num = options_num
        self.termination_regularizer = termination_regularizer
        self.ent_coff = ent_coff
        self.use_baseline = use_baseline
        self.terminal_mask = terminal_mask
        self.double_q = double_q
        self.boltzmann_temperature = boltzmann_temperature
        self.use_eps_greedy = use_eps_greedy

        self.q_net = TargetTwin(CriticQvalueAll(self.obs_spec,
                                                rep_net_params=self._rep_net_params,
                                                output_shape=self.options_num,
                                                network_settings=network_settings['q'])).to(self.device)

        self.intra_option_net = OcIntraOption(self.obs_spec,
                                              rep_net_params=self._rep_net_params,
                                              output_shape=self.a_dim,
                                              options_num=self.options_num,
                                              network_settings=network_settings['intra_option']).to(self.device)
        self.termination_net = CriticQvalueAll(self.obs_spec,
                                               rep_net_params=self._rep_net_params,
                                               output_shape=self.options_num,
                                               network_settings=network_settings['termination'],
                                               out_act='sigmoid').to(self.device)

        if self.is_continuous:
            # https://discuss.pytorch.org/t/valueerror-cant-optimize-a-non-leaf-tensor/21751
            # https://blog.csdn.net/nkhgl/article/details/100047276
            self.log_std = t.as_tensor(np.full(
                (self.options_num, self.a_dim), -0.5)).requires_grad_().to(self.device)  # [P, A]
            self.intra_option_oplr = OPLR(
                [self.intra_option_net, self.log_std], intra_option_lr, clipvalue=5.)
        else:
            self.intra_option_oplr = OPLR(
                self.intra_option_net, intra_option_lr, clipvalue=5.)
        self.q_oplr = OPLR(self.q_net, q_lr, clipvalue=5.)
        self.termination_oplr = OPLR(
            self.termination_net, termination_lr, clipvalue=5.)

        self._trainer_modules.update(q_net=self.q_net,
                                     intra_option_net=self.intra_option_net,
                                     termination_net=self.termination_net,
                                     q_oplr=self.q_oplr,
                                     intra_option_oplr=self.intra_option_oplr,
                                     termination_oplr=self.termination_oplr)
        self.options = self.new_options = self._generate_random_options()

    def _generate_random_options(self):
        # [B,]
        return t.tensor(np.random.randint(0, self.options_num, self.n_copys)).to(self.device)

    def episode_step(self,
                     obs: Data,
                     env_rets: Data,
                     begin_mask: np.ndarray):
        super().episode_step(obs, env_rets, begin_mask)
        self.options = self.new_options

    @iTensor_oNumpy
    def select_action(self, obs):
        q = self.q_net(obs, cell_state=self.cell_state)  # [B, P]
        self.next_cell_state = self.q_net.get_cell_state()
        pi = self.intra_option_net(
            obs, cell_state=self.cell_state)  # [B, P, A]
        beta = self.termination_net(obs, cell_state=self.cell_state)  # [B, P]
        options_onehot = t.nn.functional.one_hot(
            self.options, self.options_num).float()    # [B, P]
        options_onehot_expanded = options_onehot.unsqueeze(-1)  # [B, P, 1]
        pi = (pi * options_onehot_expanded).sum(-2)  # [B, A]
        if self.is_continuous:
            mu = pi.tanh()  # [B, A]
            log_std = self.log_std[self.options]    # [B, A]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            actions = dist.sample().clamp(-1, 1)    # [B, A]
        else:
            pi = pi / self.boltzmann_temperature    # [B, A]
            dist = td.Categorical(logits=pi)
            actions = dist.sample()   # [B, ]
        max_options = q.argmax(-1).long()  # [B, P] => [B, ]
        if self.use_eps_greedy:
            # epsilon greedy
            if self._is_train_mode and self.expl_expt_mng.is_random(self.cur_train_step):
                self.new_options = self._generate_random_options()
            else:
                self.new_options = max_options
        else:
            beta_probs = (beta * options_onehot).sum(-1)   # [B, P] => [B,]
            beta_dist = td.Bernoulli(probs=beta_probs)
            self.new_options = t.where(
                beta_dist.sample() < 1, self.options, max_options)
        return actions, Data(action=actions,
                             last_options=self.options,
                             options=self.new_options)

    def random_action(self):
        actions = super().random_action()
        self._acts_info.update(last_options=np.random.randint(0, self.options_num, self.n_copys),
                               options=np.random.randint(0, self.options_num, self.n_copys))
        return actions

    def _preprocess_BATCH(self, BATCH):  # [T, B, *]
        BATCH = super()._preprocess_BATCH(BATCH)
        BATCH.last_options = int2one_hot(BATCH.last_options, self.options_num)
        BATCH.options = int2one_hot(BATCH.options, self.options_num)
        return BATCH

    @iTensor_oNumpy
    def _train(self, BATCH):
        q = self.q_net(BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, P]
        q_next = self.q_net.t(
            BATCH.obs_, begin_mask=BATCH.begin_mask)   # [T, B, P]
        beta_next = self.termination_net(
            BATCH.obs_, begin_mask=BATCH.begin_mask)  # [T, B, P]

        qu_eval = (q * BATCH.options).sum(-1, keepdim=True)  # [T, B, 1]
        beta_s_ = (beta_next * BATCH.options).sum(-1,
                                                  keepdim=True)  # [T, B, 1]
        q_s_ = (q_next * BATCH.options).sum(-1, keepdim=True)   # [T, B, 1]
        # https://github.com/jeanharb/option_critic/blob/5d6c81a650a8f452bc8ad3250f1f211d317fde8c/neural_net.py#L94
        if self.double_q:
            q_ = self.q_net(
                BATCH.obs_, begin_mask=BATCH.begin_mask)  # [T, B, P]
            # [T, B, P] => [T, B] => [T, B, P]
            max_a_idx = t.nn.functional.one_hot(
                q_.argmax(-1), self.options_num).float()
            q_s_max = (q_next * max_a_idx).sum(-1, keepdim=True)   # [T, B, 1]
        else:
            q_s_max = q_next.max(-1, keepdim=True)[0]   # [T, B, 1]
        u_target = (1 - beta_s_) * q_s_ + beta_s_ * q_s_max   # [T, B, 1]
        qu_target = n_step_return(BATCH.reward,
                                  self.gamma,
                                  BATCH.done,
                                  u_target,
                                  BATCH.begin_mask).detach()  # [T, B, 1]
        td_error = qu_target - qu_eval     # gradient : q   [T, B, 1]
        q_loss = (td_error.square() * BATCH.get('isw', 1.0)
                  ).mean()        # [T, B, 1] => 1
        self.q_oplr.optimize(q_loss)

        q_s = qu_eval.detach()  # [T, B, 1]
        # https://github.com/jeanharb/option_critic/blob/5d6c81a650a8f452bc8ad3250f1f211d317fde8c/neural_net.py#L130
        if self.use_baseline:
            adv = (qu_target - q_s).detach()    # [T, B, 1]
        else:
            adv = qu_target.detach()    # [T, B, 1]
        # [T, B, P] => [T, B, P, 1]
        options_onehot_expanded = BATCH.options.unsqueeze(-1)
        pi = self.intra_option_net(
            BATCH.obs, begin_mask=BATCH.begin_mask)  # [T, B, P, A]
        # [T, B, P, A] => [T, B, A]
        pi = (pi * options_onehot_expanded).sum(-2)
        if self.is_continuous:
            mu = pi.tanh()  # [T, B, A]
            log_std = self.log_std[BATCH.options.argmax(-1)]  # [T, B, A]
            dist = td.Independent(td.Normal(mu, log_std.exp()), 1)
            log_p = dist.log_prob(BATCH.action).unsqueeze(-1)  # [T, B, 1]
            entropy = dist.entropy().unsqueeze(-1)  # [T, B, 1]
        else:
            pi = pi / self.boltzmann_temperature    # [T, B, A]
            log_pi = pi.log_softmax(-1)  # [T, B, A]
            entropy = -(log_pi.exp() * log_pi).sum(-1,
                                                   keepdim=True)    # [T, B, 1]
            log_p = (BATCH.action * log_pi).sum(-1,
                                                keepdim=True)    # [T, B, 1]
        pi_loss = -(log_p * adv + self.ent_coff * entropy).mean()    # 1

        beta = self.termination_net(
            BATCH.obs, begin_mask=BATCH.begin_mask)   # [T, B, P]
        beta_s = (beta * BATCH.last_options).sum(-1,
                                                 keepdim=True)   # [T, B, 1]
        if self.use_eps_greedy:
            v_s = q.max(-1, keepdim=True)[0] - \
                self.termination_regularizer   # [T, B, 1]
        else:
            v_s = (1 - beta_s) * q_s + beta_s * \
                q.max(-1, keepdim=True)[0]    # [T, B, 1]
            # v_s = q.mean(-1, keepdim=True)  # [T, B, 1]
        beta_loss = beta_s * (q_s - v_s).detach()   # [T, B, 1]
        # https://github.com/lweitkamp/option-critic-pytorch/blob/0c57da7686f8903ed2d8dded3fae832ee9defd1a/option_critic.py#L238
        if self.terminal_mask:
            beta_loss *= (1 - BATCH.done)   # [T, B, 1]
        beta_loss = beta_loss.mean()  # 1

        self.intra_option_oplr.optimize(pi_loss)
        self.termination_oplr.optimize(beta_loss)

        return td_error, dict([
            ['LEARNING_RATE/q_lr', self.q_oplr.lr],
            ['LEARNING_RATE/intra_option_lr', self.intra_option_oplr.lr],
            ['LEARNING_RATE/termination_lr', self.termination_oplr.lr],
            # ['Statistics/option', self.options[0]],
            ['LOSS/q_loss', q_loss],
            ['LOSS/pi_loss', pi_loss],
            ['LOSS/beta_loss', beta_loss],
            ['Statistics/q_option_max', q_s.max()],
            ['Statistics/q_option_min', q_s.min()],
            ['Statistics/q_option_mean', q_s.mean()]
        ])

    def _after_train(self):
        super()._after_train()
        if self.cur_train_step % self.assign_interval == 0:
            self.q_net.sync()
