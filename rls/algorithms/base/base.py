#!/usr/bin/env python3
# encoding: utf-8

import os
import json
import numpy as np
import torch as t

from typing import (Dict,
                    Callable,
                    Union,
                    List,
                    NoReturn,
                    Optional,
                    Any)
from torch.utils.tensorboard import SummaryWriter

from rls.utils.display import colorize
from rls.utils.sundry_utils import check_or_create
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)


class Base:

    def __init__(self,
                 no_save=False,
                 base_dir='',
                 device='cpu'):
        '''
        inputs:
            a_dim: action spaces
            base_dir: the directory that store data, like model, logs, and other data
        '''
        super().__init__()
        self.no_save = no_save
        self.base_dir = base_dir
        self.device = device
        logger.info(colorize(f"PyTorch Tensor Device: {self.device}"))

        self.cp_dir, self.log_dir = [os.path.join(base_dir, i) for i in ['model', 'log']]

        if not self.no_save:
            check_or_create(self.cp_dir, 'checkpoints(models)')

        self.global_step = t.tensor(0, dtype=t.int64)
        self._worker_modules = {}
        self._trainer_modules = {'global_step': self.global_step}
        self.writer = self._create_writer(self.log_dir)  # TODO: Annotation

    def _create_writer(self, log_dir: str) -> SummaryWriter:
        if not self.no_save:
            check_or_create(log_dir, 'logs(summaries)')
            return SummaryWriter(log_dir)

    def resume(self, base_dir: Optional[str] = None) -> Dict:
        """
        check whether chekpoint and model be within cp_dir, if in it, restore otherwise initialize randomly.
        """
        if base_dir:
            ckpt_path = os.path.join(base_dir, 'model/checkpoint.pth')
        else:
            ckpt_path = os.path.join(self.cp_dir, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            checkpoint = t.load(ckpt_path)
            try:
                for k, v in self._trainer_modules.items():
                    if hasattr(v, 'load_state_dict'):
                        self._trainer_modules[k].load_state_dict(checkpoint[k])
                    elif hasattr(v, 'fill_'):
                        self._trainer_modules[k].fill_(checkpoint[k])
                    else:
                        raise ValueError
            except Exception as e:
                logger.error(e)
                raise Exception(colorize(f'Resume model from {ckpt_path} FAILED.', color='red'))
            else:
                logger.info(colorize(f'Resume model from {ckpt_path} SUCCESSFULLY.', color='green'))
        else:
            logger.info(colorize('Initialize model SUCCESSFULLY.', color='green'))

        path = f'{self.base_dir}/step.json'
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        return dict(train_step=int(data.get('train_step', 0)),
                    frame_step=int(data.get('frame_step', 0)),
                    episode=int(data.get('episode', 0)))

    def save(self, **kwargs) -> NoReturn:
        """
        save the training model 
        """
        if not self.no_save:
            train_step = int(kwargs.get('train_step', 0))
            data = {}
            for k, v in self._trainer_modules.items():
                if hasattr(v, 'state_dict'):
                    data[k] = v.state_dict()
                else:
                    data[k] = v
            t.save(data, os.path.join(self.cp_dir, 'checkpoint.pth'))
            logger.info(colorize(f'Save checkpoint success. Training step: {train_step}', color='green'))

            with open(f'{self.base_dir}/step.json', 'w') as f:
                json.dump(kwargs, f)

    def write_summaries(self,
                        global_step: Union[int, t.Tensor],
                        summaries: Dict = {},
                        writer: Optional[SummaryWriter] = None) -> NoReturn:
        '''
        write tf summaries showing in tensorboard.
        '''
        if not self.no_save:
            writer = writer or self.writer
            for k, v in summaries.items():
                writer.add_scalar(k, v, global_step=global_step)

    def get_worker_params(self):
        pass

    def set_worker_params(self, weights_list):
        pass

    def save_weights(self, path: str) -> Any:
        """
        save trained weights
        :return: None
        """
        # self.net.save_weights(os.path.join(path, 'net.ckpt'))
        pass

    def load_weights(self, path: str) -> Any:
        """
        load trained weights
        :return: None
        """
        # self.net.load_weights(os.path.join(path, 'net.ckpt'))
        pass

    def close(self) -> Any:
        """
        end training, and export the training model
        """
        pass

    def get_global_step(self) -> int:
        """
        get the current training step.
        """
        return self.global_step

    def set_global_step(self, num: int) -> NoReturn:
        """
        set the start training step.
        """
        self.global_step = t.tensor(num, dtype=t.int64)  # TODO
