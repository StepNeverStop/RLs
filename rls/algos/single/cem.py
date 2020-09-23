#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf

from rls.nn.layers import mlp
from rls.algos.base.on_policy import make_on_policy_class


class Model(tf.keras.Model):

    def __init__(self, vector_dim, output_shape, hidden_units, is_continuous):
        super().__init__()
        self.is_continuous = is_continuous
        out_activation = 'tanh' if self.is_continuous else None
        self.net = mlp(hidden_units, act_fn='tanh', output_shape=output_shape, out_activation=out_activation, out_layer=True)
        self.weights_2dim = [[i, j] for i, j in zip([vector_dim] + hidden_units, hidden_units + [output_shape])]
        self.weights_nums = np.asarray(self.weights_2dim).prod(axis=-1).tolist()
        self.weights_total_nums = np.asarray(self.weights_2dim).prod(axis=-1).sum() + np.asarray(hidden_units).sum() + output_shape
        self(tf.keras.Input(shape=vector_dim))  # 初始化网络权重

    @tf.function
    def call(self, s):
        if self.is_continuous:
            return self.net(s)
        else:
            return tf.argmax(self.net(s), axis=-1)

    def set_wb(self, weights):
        start = 0
        wbs = []
        for dim_list, nums in zip(self.weights_2dim, self.weights_nums):
            w = weights[start:start + nums].reshape(dim_list)
            b = weights[start + nums:start + nums + dim_list[-1]]
            wbs.append(w)
            wbs.append(b)
            start += nums + dim_list[-1]
        self.set_weights(wbs)


class CEM(make_on_policy_class(mode='share')):
    '''
    Cross-Entropy Method
    '''

    def __init__(self,
                 envspec,

                 hidden_units=[32, 32],
                 frac=0.2,
                 init_var=1,
                 extra_std=1,
                 extra_decay_eps=200,
                 extra_var_last_multiplier=0.2,
                 envs_per_popu=5,   # 环境数/模型数 余数为0
                 **kwargs):
        super().__init__(envspec=envspec, **kwargs)
        self.frac = frac
        self.hidden_units = hidden_units
        self.init_var = init_var
        self.extra_std = extra_std
        self.extra_decay_eps = extra_decay_eps
        self.envs_per_popu = envs_per_popu
        self.extra_var_last_multiplier = extra_var_last_multiplier

    def choose_action(self, s, visual_s, evaluation=False):
        self._check_agents(s)
        a = [model(s_).numpy() for model, s_ in zip(self.cem_models, np.split(s, self.populations, axis=0))]
        if self.is_continuous:
            a = np.vstack(a)
        else:
            a = np.hstack(a)
        return a

    @tf.function
    def _get_action(self, s, visual_s):
        s, visual_s = self._tf_data_cast(s, visual_s)
        with tf.device(self.device):
            pass

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        self.returns += r * (1 - self.dones)
        self.dones += done
        pass

    def learn(self, **kwargs):
        self.train_step = kwargs.get('train_step')
        rets = self.returns.reshape(-1, self.envs_per_popu).mean(axis=-1)
        elites_idxs = rets.argsort()[-self.n_elite:]
        elites_weights = np.array(self.models_weights)[elites_idxs, :]
        self.mu = np.mean(elites_weights, axis=0)
        self.sigma = np.var(elites_weights, axis=0)
        self._update_models_weights()
        self._reset_variables()
        self.write_training_summaries(self.train_step, dict([
            ['Statistics/mu', self.mu.mean()],
            ['Statistics/sigma', self.sigma.mean()],
            ['Statistics/sample_std', self.sample_std.mean()]
        ]))

    def _check_agents(self, s):
        '''
        用于为实例赋予种群数量属性，并且初始化变量
        params : 状态列表S，一个环境下有多少个智能体就包含多少个状态向量
        '''
        if not hasattr(self, 'populations'):
            assert s.shape[0] % self.envs_per_popu == 0, '环境数必须可以整除envs_per_popu系数'
            self.populations = int(s.shape[0] / self.envs_per_popu)
            self._build()

    def _build(self):
        '''
        构建实体模型，初始化变量
        '''
        self.n_elite = max(int(np.round(self.populations * self.frac)), 1)
        self.cem_models = [Model(self.s_dim, self.a_dim, self.hidden_units, self.is_continuous) for i in range(self.populations)]
        self.mu = np.random.randn(self.cem_models[0].weights_total_nums)
        self.sigma = np.ones(self.cem_models[0].weights_total_nums) * self.init_var
        self._update_models_weights()
        self._reset_variables()

    def _reset_variables(self):
        '''
        初始化return列表和done标志列表
        '''
        self.returns = np.zeros(self.populations * self.envs_per_popu, dtype=np.float32)
        self.dones = np.full(self.populations * self.envs_per_popu, False)

    def _update_models_weights(self):
        '''
        重新给模型赋参数
        '''
        extra_var_multiplier = max((1.0 - self.train_step / self.extra_decay_eps), self.extra_var_last_multiplier)
        self.sample_std = np.sqrt(self.sigma + np.square(self.extra_std) * extra_var_multiplier)
        self.models_weights = [self.mu + self.sample_std * np.random.randn(self.mu.shape[0]) for i in range(self.populations)]
        [m.set_wb(wb) for m, wb in zip(self.cem_models, self.models_weights)]
