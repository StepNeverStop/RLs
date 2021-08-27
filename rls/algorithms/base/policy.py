#!/usr/bin/env python3
# encoding: utf-8

import os
import sys
import numpy as np
import torch as t

from typing import (Dict,
                    Callable,
                    Union,
                    List,
                    Tuple,
                    NoReturn,
                    Optional,
                    Any)
from torch.utils.tensorboard import SummaryWriter

from rls.algorithms.base.base import Base
from rls.common.specs import Data
from rls.utils.display import colorize
from rls.utils.sundry_utils import check_or_create
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


class Policy(Base):

    def __init__(self,
                 n_copys=1,
                 no_save=False,
                 base_dir='',
                 device='cpu',
                 max_train_step=sys.maxsize,

                 max_frame_step=sys.maxsize,
                 max_train_episode=sys.maxsize,
                 save_frequency=100,
                 save2single_file=False,
                 gamma=0.999,
                 decay_lr=False,
                 normalize_vector_obs=False,
                 obs_with_pre_action=False,
                 rep_net_params={
                     'use_encoder': False,
                     'use_rnn': False,  # always false, using -r to active RNN
                     'vector_net_params': {
                         'network_type': 'adaptive'  # rls.nn.represents.vectors
                     },
                     'visual_net_params': {
                         'visual_feature': 128,
                         'network_type': 'simple'  # rls.nn.represents.visuals
                     },
                     'encoder_net_params': {
                         'output_dim': 16
                     },
                     'memory_net_params': {
                         'rnn_units': 16,
                         'network_type': 'lstm'
                     }},
                 **kwargs):  # TODO: remove this
        '''
        inputs:
            a_dim: action spaces
            base_dir: the directory that store data, like model, logs, and other data
        '''
        self.n_copys = n_copys
        self.no_save = no_save
        self.base_dir = base_dir
        self.device = device
        logger.info(colorize(f"PyTorch Tensor Device: {self.device}"))
        self.max_train_step = max_train_step

        self.max_frame_step = max_frame_step
        self.max_train_episode = max_train_episode
        self._save_frequency = save_frequency
        self._save2single_file = save2single_file
        self.gamma = gamma
        self._decay_lr = decay_lr    # TODO: implement
        self._normalize_vector_obs = normalize_vector_obs    # TODO: implement
        self._obs_with_pre_action = obs_with_pre_action
        self._rep_net_params = dict(rep_net_params)

        super().__init__()

        # TODO: optimization
        self.use_rnn = rep_net_params.get('use_rnn', False)
        self.memory_net_params = rep_net_params.get('memory_net_params', {
            'rnn_units': 16,
            'network_type': 'lstm'
        })

        self.cp_dir, self.log_dir = [os.path.join(
            base_dir, i) for i in ['model', 'log']]

        if not self.no_save:
            check_or_create(self.cp_dir, 'checkpoints(models)')
        self.writer = self._create_writer(self.log_dir)  # TODO: Annotation

        self.cur_interact_step = t.tensor(0).long().to(self.device)
        self.cur_train_step = t.tensor(0).long().to(self.device)
        self.cur_frame_step = t.tensor(0).long().to(self.device)
        self.cur_episode = t.tensor(0).long().to(self.device)

        self._trainer_modules = {'cur_train_step': self.cur_train_step}

    def __call__(self, obs):
        raise NotImplementedError

    def select_action(self, obs):
        raise NotImplementedError

    def random_action(self):
        raise NotImplementedError

    def setup(self, is_train_mode):
        self._is_train_mode = is_train_mode

    def episode_reset(self):
        raise NotImplementedError

    def episode_step(self):
        self.cur_interact_step += 1
        self.cur_frame_step += self.n_copys

    def episode_end(self):
        self.cur_episode += 1

    def learn(self, BATCH: Data):
        raise NotImplementedError

    def close(self):
        pass

    def save(self) -> NoReturn:
        """
        save the training model 
        """
        if not self.no_save:
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

    def _create_writer(self, log_dir: str) -> SummaryWriter:
        if not self.no_save:
            check_or_create(log_dir, 'logs(summaries)')
            return SummaryWriter(log_dir)

    def _write_train_summaries(self,
                               cur_train_step: Union[int, t.Tensor],
                               summaries: Dict = {},
                               writer: Optional[SummaryWriter] = None) -> NoReturn:
        '''
        write summaries showing in tensorboard.
        '''
        if not self.no_save:
            writer = writer or self.writer
            for k, v in summaries.items():
                writer.add_scalar(k, v, global_step=cur_train_step)

    def _initial_cell_state(self, batch: int, dtype='numpy') -> Tuple[Union[t.Tensor, np.ndarray]]:
        if self.use_rnn:
            if self.memory_net_params['network_type'] == 'lstm':
                if dtype == 'numpy':
                    return {'hx': np.zeros((batch, self.memory_net_params['rnn_units'])),
                            'cx': np.zeros((batch, self.memory_net_params['rnn_units']))}
                elif dtype == 'tensor':
                    return {'hx': t.zeros((batch, self.memory_net_params['rnn_units'])).to(self.device),
                            'cx': t.zeros((batch, self.memory_net_params['rnn_units'])).to(self.device)}
                else:
                    raise NotImplementedError
            elif self.memory_net_params['network_type'] == 'gru':
                if dtype == 'numpy':
                    return {'hx': np.zeros((batch, self.memory_net_params['rnn_units']))}
                elif dtype == 'tensor':
                    return {'hx': np.zeros((batch, self.memory_net_params['rnn_units'])).to(self.device)}
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            return None
