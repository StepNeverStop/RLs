
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, NoReturn, Optional, Union

import numpy as np
import torch as t

from rls.algorithms.base.base import Base
from rls.common.specs import Data
from rls.common.yaml_ops import load_config
from rls.utils.display import colorize
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
            logger.info(colorize(
                'using SARL algorithm to train Multi-Agent task, model has been changed to independent-SARL automatically.'))

        self.models = {}
        for id in self._agent_ids:
            _algo_args = deepcopy(algo_args)
            _algo_args.agent_id = id
            if self._n_agents > 1:
                _algo_args.base_dir += f'/i{sarl_model_class.__name__}-{id}'
            self.models[id] = sarl_model_class(
                agent_spec=agent_specs[id], **_algo_args)

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
        for id in self._agent_ids:
            self.models[id].setup(is_train_mode=is_train_mode, store=store)

    def episode_reset(self):
        # 1
        for id in self._agent_ids:
            self.models[id].episode_reset()

    def episode_step(self,
                     obs,
                     acts: Dict[str, Data],
                     env_rets: Dict[str, Data]):
        # 3
        for id in self._agent_ids:
            self.models[id].episode_step(obs[id], acts[id], env_rets[id], obs['global'].begin_mask)

    def episode_end(self):
        for id in self._agent_ids:
            self.models[id].episode_end()

    def learn(self, BATCH_DICT):
        for id in self._agent_ids:
            self.models[id].learn(BATCH_DICT[id])

    def close(self):
        for id in self._agent_ids:
            self.models[id].close()

    def save(self):
        for id in self._agent_ids:
            self.models[id].save()

    def resume(self, base_dir: Optional[str] = None) -> Dict:
        for id in self._agent_ids:
            if self._n_agents > 1 and base_dir is not None:
                base_dir += f'/i{self.models[id].__class__.__name__}-{id}'
            self.models[id].resume(base_dir)

    @property
    def still_learn(self):
        return all(model.still_learn for model in self.models.values())

    def write_recorder_summaries(self, summaries: Dict[str, Dict]) -> NoReturn:
        '''
        write summaries showing in tensorboard.
        '''
        for id in self._agent_ids:
            self.models[id].write_recorder_summaries(summaries=summaries[id])
