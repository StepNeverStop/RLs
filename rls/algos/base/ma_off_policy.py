#!/usr/bin/env python3
# encoding: utf-8

import importlib
import numpy as np
import torch as t

from abc import abstractmethod
from typing import (List,
                    Dict,
                    Union,
                    NoReturn)

from rls.algos.base.ma_policy import MultiAgentPolicy
from rls.common.yaml_ops import load_config
from rls.memories.multi_replay_buffers import MultiAgentExperienceReplay
from rls.common.specs import BatchExperiences


class MultiAgentOffPolicy(MultiAgentPolicy):
    def __init__(self, envspecs, **kwargs):
        super().__init__(envspecs=envspecs, **kwargs)

        self.buffer_size = int(kwargs.get('buffer_size', 10000))

        self.n_step = int(kwargs.get('n_step', 1))

        self.burn_in_time_step = int(kwargs.get('burn_in_time_step', 10))
        self.train_time_step = int(kwargs.get('train_time_step', 10))
        self.episode_batch_size = int(kwargs.get('episode_batch_size', 32))
        self.episode_buffer_size = int(kwargs.get('episode_buffer_size', 10000))

        self.train_times_per_step = int(kwargs.get('train_times_per_step', 1))

    def initialize_data_buffer(self) -> NoReturn:
        '''
        TODO: Annotation
        '''
        _buffer_args = {}
        # if self.use_rnn:
        #     _type = 'EpisodeExperienceReplay'
        #     _buffer_args.update(
        #         batch_size=self.episode_batch_size,
        #         capacity=self.episode_buffer_size,
        #         burn_in_time_step=self.burn_in_time_step,
        #         train_time_step=self.train_time_step,
        #         n_copys=self.n_copys
        #     )
        # else:
        _type = 'ExperienceReplay'
        _buffer_args.update(
            batch_size=self.batch_size,
            capacity=self.buffer_size
        )
        # if self.use_priority:
        #     raise NotImplementedError("multi agent algorithms now not support prioritized experience replay.")
        if self.n_step > 1:
            _type = 'NStep' + _type
            _buffer_args.update(
                n_step=self.n_step,
                gamma=self.gamma,
                n_copys=self.n_copys
            )
            self.gamma = self.gamma ** self.n_step

        default_buffer_args = load_config(f'rls/configs/off_policy_buffer.yaml')['MultiAgentExperienceReplay'][_type]
        default_buffer_args.update(_buffer_args)

        self.data = MultiAgentExperienceReplay(n_agents=self.n_agents_percopy,
                                               single_agent_buffer_class=getattr(importlib.import_module(f'rls.memories.single_replay_buffers'), _type),
                                               buffer_config=default_buffer_args)

    def store_data(self, expss: List[BatchExperiences]) -> NoReturn:
        """
        for off-policy training, use this function to store <s, a, r, s_, done> into ReplayBuffer.
        """
        # self._running_average()
        self.data.add(expss)

    def no_op_store(self, expss: List[BatchExperiences]) -> NoReturn:
        # self._running_average()
        self.data.add(expss)

    def _target_params_update(self):
        pass

    def get_transitions(self) -> BatchExperiences:
        '''
        TODO: Annotation
        '''
        expss = self.data.sample()   # 经验池取数据
        return self._data_process2dict(expss)

    def _data_process2dict(self, expss: List[BatchExperiences]) -> List[BatchExperiences]:
        # TODO 优化
        rets = []
        for i, exps in enumerate(expss):
            if not self.envspecs[i].is_continuous:
                exps.action = int2one_hot(exps.action.astype(np.int32), self.envspecs[i].a_dim)
            rets.append(exps)
            # exps.obs.vector = self.normalize_vector_obs()
            # exps.obs_.vector = self.normalize_vector_obs()
        return rets

    @abstractmethod
    def _train(self, *args):
        '''
        NOTE: usually need to override this function
        TODO: Annotation
        '''
        return None

    def _learn(self, function_dict: Dict = {}) -> NoReturn:
        '''
        TODO: Annotation
        '''

        if self.data.can_sample:
            self.intermediate_variable_reset()
            data = self.get_transitions()

            # --------------------------------------训练主程序，返回可能用于PER权重更新的TD error，和需要输出tensorboard的信息
            summaries = self._train(data)
            # --------------------------------------

            # --------------------------------------target网络的更新部分
            self._target_params_update()
            # --------------------------------------

            for k, v in summaries.items():
                self.summaries[k].update(v)

            # --------------------------------------写summary到tensorboard
            self.write_training_summaries(self.global_step, self.summaries)
            # --------------------------------------
