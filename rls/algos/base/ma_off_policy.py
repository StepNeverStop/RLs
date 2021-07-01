#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import tensorflow as tf

from abc import abstractmethod
from typing import (List,
                    Dict,
                    Union,
                    NoReturn)

from rls.algos.base.ma_policy import MultiAgentPolicy
from rls.common.yaml_ops import load_yaml
from rls.memories.multi_replay_buffers import MultiAgentExperienceReplay
from rls.utils.specs import (BatchExperiences,
                             NamedTupleStaticClass)


class MultiAgentOffPolicy(MultiAgentPolicy):
    def __init__(self, envspecs, **kwargs):
        super().__init__(envspecs=envspecs, **kwargs)

        self.buffer_size = int(kwargs.get('buffer_size', 10000))
        self.n_step = int(kwargs.get('n_step', 1))
        self.train_times_per_step = int(kwargs.get('train_times_per_step', 1))

    def initialize_data_buffer(self) -> NoReturn:
        '''
        TODO: Annotation
        '''
        _buffer_args = dict(n_agents=self.n_agents_percopy, batch_size=self.batch_size, capacity=self.buffer_size)
        default_buffer_args = load_yaml(f'rls/configs/off_policy_buffer.yaml')['MultiAgentExperienceReplay']
        default_buffer_args.update(_buffer_args)
        self.data = MultiAgentExperienceReplay(**default_buffer_args)

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
                assert 'action' in exps._fields, "assert 'action' in exps._fields"
                exps = exps._replace(action=int2one_hot(exps.action.astype(np.int32), self.envspecs[i].a_dim))
            assert 'obs' in exps._fields and 'obs_' in exps._fields, "'obs' in exps._fields and 'obs_' in exps._fields"
            rets.append(NamedTupleStaticClass.data_convert(self.data_convert, exps))
        # exps = exps._replace(
        #     obs=exps.obs._replace(vector=self.normalize_vector_obs()),
        #     obs_=exps.obs_._replace(vector=self.normalize_vector_obs()))
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

        if self.data.is_lg_batch_size:
            self.intermediate_variable_reset()
            data = self.get_transitions()

            # --------------------------------------训练主程序，返回可能用于PER权重更新的TD error，和需要输出tensorboard的信息
            summaries = self._train(data)
            # --------------------------------------

            # --------------------------------------target网络的更新部分
            self._target_params_update()
            # --------------------------------------

            # --------------------------------------写summary到tensorboard
            self.write_training_summaries(self.global_step, summaries)
            # --------------------------------------
