#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
from typing import Dict, List, NoReturn, Optional, Union

import numpy as np
import torch as th

from rls.algorithms.base.base import Base
from rls.common.data import Data
from rls.common.when import Every, Until
from rls.utils.display import colorize
from rls.utils.logging_utils import get_logger
from rls.utils.summary_collector import SummaryCollector
from rls.utils.sundry_utils import check_or_create

logger = get_logger(__name__)


class Policy(Base):

    def __init__(self,
                 n_copies=1,
                 is_save=True,
                 base_dir='',
                 device: str = 'cpu',
                 max_train_step=sys.maxsize,

                 max_frame_step=sys.maxsize,
                 max_train_episode=sys.maxsize,
                 save_frequency=100,
                 save2single_file=False,
                 n_step_value=4,
                 gamma=0.999,
                 logger_types=['none'],
                 decay_lr=False,
                 normalize_vector_obs=False,
                 obs_with_pre_action=False,
                 oplr_params=dict(),
                 rep_net_params={
                     'vector_net_params': {
                         'h_dim': 16,
                         'network_type': 'adaptive'  # rls.nn.represents.vectors
                     },
                     'visual_net_params': {
                         'h_dim': 128,
                         'network_type': 'simple'  # rls.nn.represents.visuals
                     },
                     'encoder_net_params': {
                         'h_dim': 16,
                         'network_type': 'identity'  # rls.nn.represents.encoders
                     },
                     'memory_net_params': {
                         'rnn_units': 16,
                         'network_type': 'lstm'
                     }},
                 **kwargs):
        """
        inputs:
            a_dim: action spaces
            base_dir: the directory that store data, like model, logs, and other data
        """
        self.n_copies = n_copies
        self._is_save = is_save
        self._base_dir = base_dir
        self._training_name = os.path.split(self._base_dir)[-1]
        self.device = device
        logger.info(colorize(f"PyTorch Tensor Device: {self.device}"))
        self._max_train_step = max_train_step

        self._should_learn_cond_train_step = Until(max_train_step)
        self._should_learn_cond_frame_step = Until(max_frame_step)
        self._should_learn_cond_train_episode = Until(max_train_episode)
        self._should_save_model = Every(save_frequency)

        self._save2single_file = save2single_file
        self.gamma = gamma
        self._logger_types = logger_types
        self._n_step_value = n_step_value
        self._decay_lr = decay_lr  # TODO: implement
        self._normalize_vector_obs = normalize_vector_obs  # TODO: implement
        self._obs_with_pre_action = obs_with_pre_action
        self._rep_net_params = dict(rep_net_params)
        self._oplr_params = dict(oplr_params)

        super().__init__()

        self.memory_net_params = rep_net_params.get('memory_net_params', {
            'rnn_units': 16,
            'network_type': 'lstm'
        })
        self.use_rnn = self.memory_net_params.get(
            'network_type', 'identity') != 'identity'

        self.cp_dir, self.log_dir = [os.path.join(
            base_dir, i) for i in ['model', 'log']]

        if self._is_save:
            check_or_create(self.cp_dir, 'checkpoints(models)')

        self._cur_interact_step = th.tensor(0).long().to(self.device)
        self._cur_train_step = th.tensor(0).long().to(self.device)
        self._cur_frame_step = th.tensor(0).long().to(self.device)
        self._cur_episode = th.tensor(0).long().to(self.device)

        self._trainer_modules = {
            '_cur_train_step': self._cur_train_step,
            '_cur_frame_step': self._cur_frame_step,
            '_cur_episode': self._cur_episode
        }

        self._buffer = self._build_buffer()
        self._loggers = self._build_loggers() if self._is_save else list()
        self._summary_collector = self._build_summary_collector()

    def __call__(self, obs):
        raise NotImplementedError

    def select_action(self, obs):
        raise NotImplementedError

    def random_action(self):
        raise NotImplementedError

    def setup(self, is_train_mode=True, store=True):
        self._is_train_mode = is_train_mode
        self._store = store

    def episode_reset(self):
        raise NotImplementedError

    def episode_step(self):
        if self._is_train_mode:
            self._cur_interact_step += 1
            self._cur_frame_step += self.n_copies

    def episode_end(self):
        if self._is_train_mode:
            self._cur_episode += 1

    def learn(self, BATCH: Data):
        raise NotImplementedError

    def close(self):
        pass

    def save(self) -> NoReturn:
        """
        save the training model 
        """
        if self._is_save:
            _data = {}
            for k, v in self._trainer_modules.items():
                if hasattr(v, 'state_dict'):
                    _data[k] = v.state_dict()
                else:
                    _data[k] = v  # tensor/Number
            if self._save2single_file:
                th.save(_data, os.path.join(self.cp_dir, 'checkpoint.pth'))
            else:
                for k, v in _data.items():
                    th.save(v, os.path.join(self.cp_dir, f'{k}.pth'))
            logger.info(colorize(f'Save checkpoint success. Training step: {self._cur_train_step}', color='green'))

    def resume(self, base_dir: Optional[str] = None):
        """
        check whether chekpoint and model be within cp_dir, if in it, restore otherwise initialize randomly.
        """
        cp_dir = os.path.join(base_dir or self._base_dir, 'model')
        if self._save2single_file:
            ckpt_path = os.path.join(cp_dir, 'checkpoint.pth')
            if os.path.exists(ckpt_path):
                checkpoint = th.load(ckpt_path, map_location=self.device)
                for k, v in self._trainer_modules.items():
                    if hasattr(v, 'load_state_dict'):
                        self._trainer_modules[k].load_state_dict(checkpoint[k])
                    else:
                        getattr(self, k).fill_(checkpoint[k])
                logger.info(
                    colorize(f'Resume model from {ckpt_path} SUCCESSFULLY.', color='green'))
        else:
            for k, v in self._trainer_modules.items():
                model_path = os.path.join(cp_dir, f'{k}.pth')
                if os.path.exists(model_path):
                    if hasattr(v, 'load_state_dict'):
                        self._trainer_modules[k].load_state_dict(th.load(model_path, map_location=self.device))
                    else:
                        getattr(self, k).fill_(th.load(model_path, map_location=self.device))
                    logger.info(colorize(f'Resume model from {model_path} SUCCESSFULLY.', color='green'))

    @property
    def still_learn(self):
        return self._should_learn_cond_train_step(self._cur_train_step) \
               and self._should_learn_cond_frame_step(self._cur_frame_step) \
               and self._should_learn_cond_train_episode(self._cur_episode)

    def write_log(self,
                  log_step: Union[int, th.Tensor] = None,
                  summaries: Dict = {},
                  step_type: str = None):
        self._write_log(log_step, summaries, step_type)

    # customed

    def _build_buffer(self):
        raise NotImplementedError

    def _build_loggers(self):
        raise NotImplementedError

    def _build_summary_collector(self):
        return SummaryCollector(mode=SummaryCollector.ALL)

    def _write_log(self,
                   log_step: Union[int, th.Tensor] = None,
                   summaries: Dict = None,
                   step_type: str = None):
        assert step_type is not None or log_step is not None, 'assert step_type is not None or log_step is not None'
        if log_step is None:
            if step_type == 'step':
                log_step = self._cur_train_step
            elif step_type == 'episode':
                log_step = self._cur_episode
            elif log_step == 'frame':
                log_step = self._cur_frame_step
            else:
                raise NotImplementedError("log_step must be in ['step', 'episode', 'frame'] for now.")
        summaries = summaries or self._summary_collector.fetch()
        for logger in self._loggers:
            logger.write(summaries=summaries, step=log_step)

    def _initial_rnncs(self, batch: int, rnn_units: int = None, keys: Optional[List[str]] = None) -> Dict[
        str, np.ndarray]:
        rnn_units = rnn_units or self.memory_net_params['rnn_units']
        keys = keys or (
            ['hx', 'cx'] if self.memory_net_params['network_type'] == 'lstm' else ['hx'])
        return {k: np.zeros((batch, rnn_units)) for k in keys}
