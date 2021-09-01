#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
from typing import Any, Callable, Dict, List, NoReturn, Optional, Tuple, Union

import numpy as np
import torch as t
from torch.utils.tensorboard import SummaryWriter

from rls.algorithms.base.base import Base
from rls.common.specs import Data
from rls.utils.display import colorize
from rls.utils.logging_utils import get_logger
from rls.utils.sundry_utils import check_or_create

logger = get_logger(__name__)


class Policy(Base):

    def __init__(self,
                 n_copys=1,
                 is_save=True,
                 base_dir='',
                 device='cpu',
                 max_train_step=sys.maxsize,

                 max_frame_step=sys.maxsize,
                 max_train_episode=sys.maxsize,
                 save_frequency=100,
                 save2single_file=False,
                 n_step_value=4,
                 gamma=0.999,
                 decay_lr=False,
                 normalize_vector_obs=False,
                 obs_with_pre_action=False,
                 optim_params=dict(),
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
        '''
        inputs:
            a_dim: action spaces
            base_dir: the directory that store data, like model, logs, and other data
        '''
        self.n_copys = n_copys
        self._is_save = is_save
        self.base_dir = base_dir
        self.device = device
        logger.info(colorize(f"PyTorch Tensor Device: {self.device}"))
        self.max_train_step = max_train_step

        self.max_frame_step = max_frame_step
        self.max_train_episode = max_train_episode
        self._save_frequency = save_frequency
        self._save2single_file = save2single_file
        self.gamma = gamma
        self._n_step_value = n_step_value
        self._decay_lr = decay_lr    # TODO: implement
        self._normalize_vector_obs = normalize_vector_obs    # TODO: implement
        self._obs_with_pre_action = obs_with_pre_action
        self._rep_net_params = dict(rep_net_params)
        self._optim_params = dict(optim_params)

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
        self.writer = self._create_writer(self.log_dir)  # TODO: Annotation

        self.cur_interact_step = t.tensor(0).long().to(self.device)
        self.cur_train_step = t.tensor(0).long().to(self.device)
        self.cur_frame_step = t.tensor(0).long().to(self.device)
        self.cur_episode = t.tensor(0).long().to(self.device)

        self._trainer_modules = {'cur_train_step': self.cur_train_step}

        self._buffer = self._build_buffer()

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
            self.cur_interact_step += 1
            self.cur_frame_step += self.n_copys

    def episode_end(self):
        if self._is_train_mode:
            self.cur_episode += 1

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
                t.save(_data, os.path.join(self.cp_dir, 'checkpoint.pth'))
            else:
                for k, v in _data.items():
                    t.save(v, os.path.join(self.cp_dir, f'{k}.pth'))
            logger.info(colorize(
                f'Save checkpoint success. Training step: {self.cur_train_step}', color='green'))

    def resume(self, base_dir: Optional[str] = None) -> Dict:
        """
        check whether chekpoint and model be within cp_dir, if in it, restore otherwise initialize randomly.
        """
        cp_dir = os.path.join(base_dir or self.base_dir, 'model')
        if self._save2single_file:
            ckpt_path = os.path.join(cp_dir, 'checkpoint.pth')
            if os.path.exists(ckpt_path):
                checkpoint = t.load(ckpt_path)
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
                        self._trainer_modules[k].load_state_dict(
                            t.load(model_path))
                    else:
                        getattr(self, k).fill_(t.load(model_path))
                    logger.info(
                        colorize(f'Resume model from {model_path} SUCCESSFULLY.', color='green'))

    @property
    def still_learn(self):
        return self.cur_train_step < self.max_train_step \
            and self.cur_frame_step < self.max_frame_step \
            and self.cur_episode < self.max_train_episode

    def write_recorder_summaries(self, summaries):
        raise NotImplementedError

    # customed

    def _build_buffer(self):
        raise NotImplementedError

    def _create_writer(self, log_dir: str) -> SummaryWriter:
        if self._is_save:
            check_or_create(log_dir, 'logs(summaries)')
            return SummaryWriter(log_dir)

    def _write_train_summaries(self,
                               cur_train_step: Union[int, t.Tensor],
                               summaries: Dict = {},
                               writer: Optional[SummaryWriter] = None) -> NoReturn:
        '''
        write summaries showing in tensorboard.
        '''
        if self._is_save:
            writer = writer or self.writer
            for k, v in summaries.items():
                writer.add_scalar(k, v, global_step=cur_train_step)

    def _initial_cell_state(self, batch: int, rnn_units: int = None, keys: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        rnn_units = rnn_units or self.memory_net_params['rnn_units']
        keys = keys or (
            ['hx', 'cx'] if self.memory_net_params['network_type'] == 'lstm' else ['hx'])
        return {k: np.zeros((batch, rnn_units)) for k in keys}
