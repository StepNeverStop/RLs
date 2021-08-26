
import torch as t
import numpy as np

from copy import deepcopy
from typing import (Dict,
                    Callable,
                    Union,
                    List,
                    NoReturn,
                    Optional,
                    Any)
from collections import defaultdict

from rls.algorithms.base.base import Base
from rls.common.specs import Data
from rls.utils.display import colorize
from rls.common.yaml_ops import load_config
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


class IndependentMA(Base):

    def __init__(self,
                 sarl_model_class,
                 agent_specs,
                 algo_args):
        super().__init__()
        self.policy_mode = sarl_model_class.policy_mode
        self.algo_args = algo_args

        self._agent_ids = list(agent_specs.keys())
        self._n_agents = len(self._agent_ids)
        if self._n_agents > 1:
            logger.info(colorize('using SARL algorithm to train Multi-Agent task, model has been changed to independent-SARL automatically.'))

        self.models = {}
        for id in self._agent_ids:
            _algo_args = deepcopy(algo_args)
            if self._n_agents > 1:
                _algo_args.base_dir += f'/i{sarl_model_class.__name__}-{id}'
            self.models[id] = sarl_model_class(agent_spec=agent_specs[id], **_algo_args)

        self._buffer = self._build_buffer()

    def _build_buffer(self):
        if self.policy_mode == 'on-policy':
            from rls.memories.onpolicy_buffer import OnPolicyDataBuffer
            buffer = OnPolicyDataBuffer(n_copys=self.algo_args.n_copys,
                                        batch_size=self.algo_args.batch_size,
                                        buffer_size=self.algo_args.buffer_size,
                                        time_step=self.algo_args.n_time_step)
        else:
            if self.algo_args.use_priority == True:
                from rls.memories.per_buffer import PrioritizedDataBuffer
                buffer = PrioritizedDataBuffer(n_copys=self.algo_args.n_copys,
                                               batch_size=self.algo_args.batch_size,
                                               buffer_size=self.algo_args.buffer_size,
                                               time_step=self.algo_args.n_time_step,
                                               max_train_step=self.algo_args.max_train_step,
                                               **load_config(f'rls/configs/buffer/off_policy_buffer.yaml')['PrioritizedDataBuffer'])
            else:
                from rls.memories.er_buffer import DataBuffer
                buffer = DataBuffer(n_copys=self.algo_args.n_copys,
                                    batch_size=self.algo_args.batch_size,
                                    buffer_size=self.algo_args.buffer_size,
                                    time_step=self.algo_args.n_time_step,)
        return buffer

    def __call__(self, obs):
        # 2
        acts = {}
        for id in self._agent_ids:
            acts[id] = self.models[id](obs[id])
        return acts

    def random_action(self):
        acts = {}
        for id in self._agent_ids:
            acts[id] = self.models[id].random_action()
        return acts

    def setup(self, is_train_mode=True, store=True):
        # 0
        self._is_train_mode = is_train_mode
        self._store = store
        for id in self._agent_ids:
            self.models[id].setup(is_train_mode=is_train_mode)

    def episode_reset(self):
        # 1
        for id in self._agent_ids:
            self.models[id].episode_reset()

    def episode_step(self,
                     obs,
                     acts: Dict[str, Dict[str, np.ndarray]],
                     env_rets: Dict[str, Data]):
        # 3
        if self._store:
            expss = {}
            for id in self._agent_ids:
                expss[id] = Data(obs=obs[id],
                                 reward=env_rets[id].reward[:, np.newaxis],  # [B, ] => [B, 1]
                                 obs_=env_rets[id].obs,
                                 done=env_rets[id].done[:, np.newaxis])
                expss[id].update(acts[id])
            expss['global'] = Data(begin_mask=obs['global'].begin_mask)
            self._buffer.add(expss)

        if self._is_train_mode \
            and self.policy_mode == 'off-policy' \
                and self._buffer.can_sample:
            rets = self.learn(self._buffer.sample())
            if self.algo_args.use_priority:
                self._buffer.update(sum(rets.values())/len(self._agent_ids))   # td_error   [T, B, 1]

        for id in self._agent_ids:
            self.models[id].episode_step(env_rets[id].done)

    def episode_end(self):
        if self._is_train_mode \
            and self.policy_mode == 'on-policy' \
                and self._buffer.can_sample:
            self.learn(self._buffer.all_data())   # on-policy replay buffer
            self._buffer.clear()

        for id in self._agent_ids:
            self.models[id].episode_end()

    def learn(self, BATCH_DICT):
        rets = {}
        for id in self._agent_ids:
            BATCH_DICT[id].begin_mask = BATCH_DICT['global'].begin_mask
            rets[id] = self.models[id].learn(BATCH_DICT[id])
        return rets

    def close(self):
        for id in self._agent_ids:
            self.models[id].close()

    def save(self):
        for id in self._agent_ids:
            self.models[id].save()

    def resume(self, base_dir: Optional[str] = None) -> Dict:
        for id in self._agent_ids:
            if self._n_agents > 1 and base_dir is not None:
                base_dir += f'/i{model.__class__.__name__}-{id}'
            self.models[id].resume(base_dir)

    @property
    def still_learn(self):
        return self.models[self._agent_ids[0]].still_learn

    def write_recorder_summaries(self, summaries: Dict[str, Dict]) -> NoReturn:
        '''
        write summaries showing in tensorboard.
        '''
        for id in self._agent_ids:
            self.models[id].write_recorder_summaries(summaries=summaries[id])
