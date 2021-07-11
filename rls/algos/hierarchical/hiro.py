#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import torch as t

from copy import deepcopy
from torch import distributions as td
from dataclasses import dataclass

from rls.nn.noised_actions import ClippedNormalNoisedAction
from rls.algos.base.off_policy import Off_Policy
from rls.memories.single_replay_buffers import ExperienceReplay
from rls.utils.np_utils import int2one_hot
from rls.utils.torch_utils import (sync_params_pairs,
                                   q_target_func)
from rls.common.specs import (BatchExperiences,
                              ModelObservations,
                              Data)
from rls.nn.models import (ActorDPG,
                           ActorDct,
                           CriticQvalueOne)
from rls.nn.utils import OPLR
from rls.common.decorator import iTensor_oNumpy


@dataclass(eq=False)
class Low_BatchExperiences(BatchExperiences):
    subgoal: np.ndarray
    next_subgoal: np.ndarray


@dataclass(eq=False)
class High_BatchExperiences(Data):
    obs: ModelObservations
    action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    subgoal: np.ndarray
    obs_: ModelObservations


class HIRO(Off_Policy):
    '''
    Data-Efficient Hierarchical Reinforcement Learning, http://arxiv.org/abs/1805.08296
    '''

    def __init__(self,
                 envspec,

                 ployak=0.995,
                 discrete_tau=1.0,
                 high_scale=1.0,
                 reward_scale=1.0,
                 sample_g_nums=100,
                 sub_goal_steps=10,
                 fn_goal_dim=0,
                 intrinsic_reward_mode='os',
                 high_batch_size=256,
                 high_buffer_size=100000,
                 low_batch_size=8,
                 low_buffer_size=10000,
                 high_actor_lr=1.0e-4,
                 high_critic_lr=1.0e-3,
                 low_actor_lr=1.0e-4,
                 low_critic_lr=1.0e-3,
                 network_settings={
                     'high_actor': [64, 64],
                     'high_critic': [64, 64],
                     'low_actor': [64, 64],
                     'low_critic': [64, 64]
                 },
                 **kwargs):
        assert not envspec.obs_spec.has_visual_observation, 'HIRO doesn\'t support visual inputs.'
        super().__init__(envspec=envspec, **kwargs)

        self.concat_vector_dim = self.obs_spec.total_vector_dim
        self.data_high = ExperienceReplay(high_batch_size, high_buffer_size)
        self.data_low = ExperienceReplay(low_batch_size, low_buffer_size)

        self.ployak = ployak
        self.discrete_tau = discrete_tau
        self.reward_scale = reward_scale
        self.fn_goal_dim = fn_goal_dim
        self.sample_g_nums = sample_g_nums
        self.sub_goal_steps = sub_goal_steps
        self.sub_goal_dim = self.concat_vector_dim - self.fn_goal_dim
        self.high_scale = np.array(
            high_scale if isinstance(high_scale, list) else [high_scale] * self.sub_goal_dim,
            dtype=np.float32)

        self.high_noised_action = ClippedNormalNoisedAction(mu=np.zeros(self.sub_goal_dim), sigma=self.high_scale * np.ones(self.sub_goal_dim),
                                                            action_bound=self.high_scale, noise_bound=self.high_scale / 2)
        self.low_noised_action = ClippedNormalNoisedAction(mu=np.zeros(self.a_dim), sigma=1.0 * np.ones(self.a_dim), noise_bound=0.5)

        self.high_actor = ActorDPG(vector_dim=self.concat_vector_dim,
                                   output_shape=self.sub_goal_dim,
                                   network_settings=network_settings['high_actor'])
        self.high_critic = CriticQvalueOne(vector_dim=self.concat_vector_dim,
                                           action_dim=self.sub_goal_dim,
                                           network_settings=network_settings['high_critic'])
        self.high_critic2 = CriticQvalueOne(vector_dim=self.concat_vector_dim,
                                            action_dim=self.sub_goal_dim,
                                            network_settings=network_settings['high_critic'])
        self.high_actor_target = deepcopy(self.high_actor)
        self.high_actor_target.eval()
        self.high_critic_target = deepcopy(self.high_critic)
        self.high_critic_target.eval()
        self.high_critic2_target = deepcopy(self.high_critic2)
        self.high_critic2_target.eval()

        if self.is_continuous:
            self.low_actor = ActorDPG(vector_dim=self.concat_vector_dim + self.sub_goal_dim,
                                      output_shape=self.a_dim,
                                      network_settings=network_settings['low_actor'])
        else:
            self.low_actor = ActorDct(vector_dim=self.concat_vector_dim + self.sub_goal_dim,
                                      output_shape=self.a_dim,
                                      network_settings=network_settings['low_actor'])
            self.gumbel_dist = td.gumbel.Gumbel(0, 1)
        self.low_critic = CriticQvalueOne(vector_dim=self.concat_vector_dim + self.sub_goal_dim,
                                          action_dim=self.a_dim,
                                          network_settings=network_settings['low_critic'])
        self.low_critic2 = CriticQvalueOne(vector_dim=self.concat_vector_dim + self.sub_goal_dim,
                                           action_dim=self.a_dim,
                                           network_settings=network_settings['low_critic'])
        self.low_actor_target = deepcopy(self.low_actor)
        self.low_actor_target.eval()
        self.low_critic_target = deepcopy(self.low_critic)
        self.low_critic_target.eval()
        self.low_critic2_target = deepcopy(self.low_critic2)
        self.low_critic2_target.eval()

        self._high_pairs = [(self.high_critic_target, self.high_critic),
                            (self.high_critic2_target, self.high_critic2)
                            (self.high_actor_target, self.high_actor)]
        self._low_pairs = [(self.low_critic_target, self.low_critic),
                           (self.low_critic2_target, self.low_critic2)
                           (self.low_actor_target, self.low_actor)]
        sync_params_pairs(self._high_pairs)
        sync_params_pairs(self._low_pairs)

        self.low_actor_oplr = OPLR(self.low_actor, low_actor_lr)
        self.low_critic_oplr = OPLR([self.low_critic, self.low_critic2], low_critic_lr)

        self.high_actor_oplr = OPLR(self.high_actor, high_actor_lr)
        self.high_critic_oplr = OPLR([self.high_critic, self.high_critic2], high_critic_lr)

        self.counts = 0
        self._high_s = [[] for _ in range(self.n_copys)]
        self._noop_subgoal = np.random.uniform(-self.high_scale, self.high_scale, size=(self.n_copys, self.sub_goal_dim))
        self.get_ir = self.generate_ir_func(mode=intrinsic_reward_mode)

        self._worker_modules.update(high_actor=self.high_actor,
                                    low_actor=self.low_actor)

        self._trainer_modules.update(self._worker_modules)
        self._trainer_modules.update(high_critic=self.high_critic,
                                     high_critic2=self.high_critic2,
                                     low_critic=self.low_critic,
                                     low_critic2=self.low_ritic2,
                                     low_actor_oplr=self.low_actor_oplr,
                                     low_critic_oplr=self.low_critic_oplr,
                                     high_actor_oplr=self.high_actor_oplr,
                                     high_critic_oplr=self.high_critic_oplr)

    def generate_ir_func(self, mode='os'):
        if mode == 'os':
            return lambda last_feat, subgoal, feat: -(last_feat + subgoal - feat).norm(2, -1, keepdim=True)
        elif mode == 'cos':
            return lambda last_feat, subgoal, feat: t.nn.functional.cosine_similarity(
                feat - last_feat,
                subgoal,
                dim=-1).unsqueeze(-1)

    def store_high_buffer(self, i):
        eps_len = len(self._high_s[i])
        intervals = list(range(0, eps_len, self.sub_goal_steps))
        if len(intervals) < 1:
            return
        left = intervals[:-1]
        right = intervals[1:]
        s, r, a, g, d, s_ = [], [], [], [], [], []
        for _l, _r in zip(left, right):
            s.append(self._high_s[i][_l:_r])
            r.append(sum(self._high_r[i][_l:_r]) * self.reward_scale)
            a.append(self._high_a[i][_l:_r])
            g.append(self._subgoals[i][_l])
            d.append(self._done[i][_r - 1])
            s_.append(self._high_s_[i][_r - 1])

        right = intervals[-1]
        s.append(self._high_s[i][right:eps_len] + [self._high_s[i][-1]] * (self.sub_goal_steps + right - eps_len))
        r.append(sum(self._high_r[i][right:eps_len]))
        a.append(self._high_a[i][right:eps_len] + [self._high_a[i][-1]] * (self.sub_goal_steps + right - eps_len))
        g.append(self._subgoals[i][right])
        d.append(self._done[i][-1])
        s_.append(self._high_s_[i][-1])
        self.data_high.add(High_BatchExperiences(
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(d),
            np.array(g),
            np.array(s_)
        ))

    def reset(self):
        self.high_noised_action.reset()
        self.low_noised_action.reset()

        self._c = np.full((self.n_copys, 1), self.sub_goal_steps, np.int32)

        for i in range(self.n_copys):
            self.store_high_buffer(i)
        self._high_r = [[] for _ in range(self.n_copys)]
        self._high_a = [[] for _ in range(self.n_copys)]
        self._high_s = [[] for _ in range(self.n_copys)]
        self._subgoals = [[] for _ in range(self.n_copys)]
        self._done = [[] for _ in range(self.n_copys)]
        self._high_s_ = [[] for _ in range(self.n_copys)]

        self._new_subgoal = np.zeros((self.n_copys, self.sub_goal_dim), dtype=np.float32)

    def partial_reset(self, done):
        self._c = np.where(done[:, np.newaxis], np.full((self.n_copys, 1), self.sub_goal_steps, np.int32), self._c)
        idx = np.where(done)[0]
        for i in idx:
            self.store_high_buffer(i)
            self._high_s[i] = []
            self._high_a[i] = []
            self._high_s_[i] = []
            self._high_r[i] = []
            self._done[i] = []
            self._subgoals[i] = []

    @iTensor_oNumpy
    def _get_action(self, obs, subgoal):
        feat = t.cat([obs.flatten_vector(), subgoal], -1)
        output = self.low_actor(feat)
        if self.is_continuous:
            mu = output
            pi = self.low_noised_action(mu)
        else:
            logits = output
            mu = logits.argmax(1)
            cate_dist = td.categorical.Categorical(logits=logits)
            pi = cate_dist.sample()
        return mu, pi

    def __call__(self, obs, evaluation=False):
        self._subgoal = np.where(self._c == self.sub_goal_steps, self.get_subgoal(obs.flatten_vector()).numpy(), self._new_subgoal)
        mu, pi = self._get_action(obs, self._subgoal)
        return mu if evaluation else pi

    def get_subgoal(self, s):
        '''
        s 当前隐状态
        '''
        new_subgoal = self.high_scale * self.high_actor(s)
        new_subgoal = self.high_noised_action(new_subgoal)
        return new_subgoal

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        for i in range(self.train_times_per_step):
            if self.data_low.can_sample and self.data_high.can_sample:
                self.intermediate_variable_reset()
                low_data = self.get_transitions(self.data_low)
                high_data = self.get_transitions(self.data_high)

                summaries = self.train_low(low_data)

                self.summaries.update(summaries)
                sync_params_pairs(self._low_pairs, self.ployak)
                if self.counts % self.sub_goal_steps == 0:
                    self.counts = 0
                    high_summaries = self.train_high(high_data)
                    self.summaries.update(high_summaries)
                    sync_params_pairs(self._high_pairs, self.ployak)
                self.counts += 1
                self.summaries.update(dict([
                    ['LEARNING_RATE/low_actor_lr', self.low_actor_oplr.lr],
                    ['LEARNING_RATE/low_critic_lr', self.low_critic_oplr.lr],
                    ['LEARNING_RATE/high_actor_lr', self.high_actor_oplr.lr],
                    ['LEARNING_RATE/high_critic_lr', self.high_critic_oplr.lr]
                ]))
                self.write_training_summaries(self.global_step, self.summaries)

    @iTensor_oNumpy
    def train_low(self, BATCH: Low_BatchExperiences):
        feat = t.cat([BATCH.obs.flatten_vector(), BATCH.subgoal], -1)
        feat_ = t.cat([BATCH.obs_.flatten_vector(), BATCH.next_subgoal], -1)

        target_output = self.low_actor_target(feat_)
        if self.is_continuous:
            action_target = target_output
        else:
            target_logits = target_output
            target_cate_dist = td.categorical.Categorical(logits=target_logits)
            target_pi = target_cate_dist.sample()
            target_log_pi = target_cate_dist.log_prob(target_pi)
            action_target = t.nn.functional.one_hot(target_pi, self.a_dim).float()
        q1 = self.low_critic(feat, BATCH.action)
        q2 = self.low_critic2(feat, BATCH.action)
        q = t.minimum(q1, q2)
        q_target = t.minimum(
            self.low_critic_target(feat_, action_target),
            self.low_critic2_target(feat_, action_target),
        )
        dc_r = q_target_func(BATCH.reward,
                             self.gamma,
                             BATCH.done,
                             q_target)
        td_error1 = q1 - dc_r
        td_error2 = q2 - dc_r
        q1_loss = td_error1.square().mean()
        q2_loss = td_error2.square().mean()
        low_critic_loss = q1_loss + q2_loss
        self.low_critic_oplr.step(low_critic_loss)

        output = self.low_actor(feat)
        if self.is_continuous:
            mu = output
        else:
            logits = output
            gumbel_noise = self.gumbel_dist.sample(BATCH.action.shape)
            logp_all = logits.log_softmax(-1)
            _pi = ((logp_all + gumbel_noise) / self.discrete_tau).softmax(-1)
            _pi_true_one_hot = t.nn.functional.one_hot(_pi.argmax(-1), self.a_dim).float()
            _pi_diff = (_pi_true_one_hot - _pi).detach()
            mu = _pi_diff + _pi
        q_actor = self.low_critic(feat, mu)
        low_actor_loss = -q_actor.mean()
        self.low_actor_oplr.step(low_actor_loss)

        self.global_step.add_(1)
        return dict([
            ['LOSS/low_actor_loss', low_actor_loss],
            ['LOSS/low_critic_loss', low_critic_loss],
            ['Statistics/low_q_min', q.min()],
            ['Statistics/low_q_mean', q.mean()],
            ['Statistics/low_q_max', q.max()]
        ])

    @iTensor_oNumpy
    def train_high(self, BATCH: High_BatchExperiences):
        # BATCH.obs_ : [B, N]
        # BATCH.obs, BATCH.action [B, T, *]
        # TODO:
        batchs = BATCH.obs.shape[0]

        s = BATCH.obs[:, 0]                                # [B, N]
        true_end = (BATCH.obs_ - s)[:, self.fn_goal_dim:]
        g_dist = td.normal.Normal(loc=true_end, scale=0.5 * self.high_scale[None, :])
        ss = BATCH.obs.unsqueeze(0)  # [1, B, T, *]
        ss = ss.repeat(self.sample_g_nums, 1, 1, 1)    # [10, B, T, *]
        ss = ss.view(-1, ss.shape[-1])  # [10*B*T, *]
        aa = BATCH.action.unsqueeze(0)  # [1, B, T, *]
        aa = aa.repeat(self.sample_g_nums, 1, 1, 1)    # [10, B, T, *]
        aa = aa.view(-1, aa.shape[-1])  # [10*B*T, *]
        gs = t.cat([
            BATCH.subgoal.unsqueeze(0),
            true_end.unsqueeze(0),
            g_dist.sample([self.sample_g_nums - 2]).clamp(-self.high_scale, self.high_scale)
        ], 0)  # [10, B, N]

        all_g = gs + s[:, self.fn_goal_dim:]
        all_g = all_g.unsqueeze(2)    # [10, B, 1, N]
        all_g = all_g.repeat(1, 1, self.sub_goal_steps, 1)  # [10, B, T, N]
        all_g = all_g.view(-1, all_g.shape[-1])    # [10*B*T, N]
        all_g = all_g - ss[:, self.fn_goal_dim:]  # [10*B*T, N]
        feat = t.cat([ss, all_g], -1)  # [10*B*T, *]
        _aa = self.low_actor(feat)  # [10*B*T, A]
        if not self.is_continuous:
            _aa = t.nn.functional.one_hot(_aa.argmax(-1), self.a_dim).float()
        diff = _aa - aa
        diff = diff.view(self.sample_g_nums, batchs, self.sub_goal_steps, -1)  # [10, B, T, A]
        diff = diff.permute(1, 0, 2, 3)   # [B, 10, T, A]
        logps = -0.5 * (diff.norm(2, -1)**2).sum(-1)  # [B, 10]
        idx = logps.argmax(-1).int()
        idx = t.stack([t.arange(batchs), idx], 1)  # [B, 2]
        g = gs.permute(1, 0, 2)[list(idx.long().T)]  # [B, N]

        q1 = self.high_critic(s, g)
        q2 = self.high_critic2(s, g)
        q = t.minimum(q1, q2)

        target_sub_goal = self.high_actor_target(BATCH.obs_) * self.high_scale
        q_target = t.minimum(
            self.high_critic_target(BATCH.obs_, target_sub_goal),
            self.high_critic2_target(BATCH.obs_, target_sub_goal),
        )

        dc_r = q_target_func(BATCH.reward,
                             self.gamma,
                             BATCH.done,
                             q_target)
        td_error1 = q1 - dc_r
        td_error2 = q2 - dc_r
        q1_loss = td_error1.square().mean()
        q2_loss = td_error2.square().mean()
        high_critic_loss = q1_loss + q2_loss

        self.high_critic_oplr.step(high_critic_loss)

        mu = self.high_actor(s) * self.high_scale
        q_actor = self.high_critic(s, mu)
        high_actor_loss = -q_actor.mean()
        self.high_actor_oplr.step(high_actor_loss)

        return dict([
            ['LOSS/high_actor_loss', high_actor_loss],
            ['LOSS/high_critic_loss', high_critic_loss],
            ['Statistics/high_q_min', q.min()],
            ['Statistics/high_q_mean', q.mean()],
            ['Statistics/high_q_max', q.max()]
        ])

    def no_op_store(self, exps: BatchExperiences):
        [o.append(_s) for o, _s in zip(self._high_s, exps.obs.flatten_vector())]
        [o.append(_a) for o, _a in zip(self._high_a, exps.action)]
        [o.append(_r) for o, _r in zip(self._high_r, exps.reward)]
        [o.append(_s_) for o, _s_ in zip(self._high_s_, exps.obs_.flatten_vector())]
        [o.append(_d) for o, _d in zip(self._done, exps.done)]
        [o.append(_subgoal) for o, _subgoal in zip(self._subgoals, self._noop_subgoal)]

        ir = self.get_ir(exps.obs.flatten_vector()[:, self.fn_goal_dim:], self._noop_subgoal, exps.obs_.flatten_vector()[:, self.fn_goal_dim:])
        # subgoal = exps.obs.flatten_vector()[:, self.fn_goal_dim:] + self._noop_subgoal - exps.obs_.flatten_vector()[:, self.fn_goal_dim:]
        subgoal = np.random.uniform(-self.high_scale, self.high_scale, size=(self.n_copys, self.sub_goal_dim))

        dl = Low_BatchExperiences(*exps.astuple(), self._noop_subgoal, subgoal)
        dl.reward = ir
        self.data_low.add(dl)
        self._noop_subgoal = subgoal

    def store_data(self, exps: BatchExperiences):
        """
        for off-policy training, use this function to store <s, a, r, s_, done> into ReplayBuffer.
        """
        [o.append(_s) for o, _s in zip(self._high_s, exps.obs.flatten_vector())]
        [o.append(_a) for o, _a in zip(self._high_a, exps.action)]
        [o.append(_r) for o, _r in zip(self._high_r, exps.reward)]
        [o.append(_s_) for o, _s_ in zip(self._high_s_, exps.obs_.flatten_vector())]
        [o.append(_d) for o, _d in zip(self._done, exps.done)]
        [o.append(_subgoal) for o, _subgoal in zip(self._subgoals, self._subgoal)]

        ir = self.get_ir(exps.obs.flatten_vector()[:, self.fn_goal_dim:], self._subgoal, exps.obs_.flatten_vector()[:, self.fn_goal_dim:])
        self._new_subgoal = np.where(self._c == 1, self.get_subgoal(exps.obs_.flatten_vector()).numpy(), exps.obs.flatten_vector()
                                     [:, self.fn_goal_dim:] + self._subgoal - exps.obs_.flatten_vector()[:, self.fn_goal_dim:])

        dl = Low_BatchExperiences(*exps.astuple(), self._subgoal, self._new_subgoal)
        dl.reward = ir
        self.data_low.add(dl)

        self._c = np.where(self._c == 1, np.full((self.n_copys, 1), self.sub_goal_steps, np.int32), self._c - 1)

    def get_transitions(self, databuffer, data_name_list=['s', 'a', 'r', 's_', 'done']):
        '''
        TODO: Annotation
        '''
        exps = databuffer.sample()   # 经验池取数据
        if not self.is_continuous:
            assert 'action' in exps.__dict__.keys(), "assert 'action' in exps.__dict__.keys()"
            a = exps.action.astype(np.int32)
            pre_shape = a.shape
            a = a.reshape(-1)
            a = int2one_hot(a, self.a_dim)
            a = a.reshape(pre_shape + (-1,))
            exps.action = a
        return exps
