#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import numpy as np
import tensorflow as tf

from typing import \
    Dict, \
    Callable, \
    Union, \
    List, \
    NoReturn, \
    Optional, \
    Any

from rls.utils.tf2_utils import \
    cast2float32, \
    cast2float64
from rls.utils.tf2_utils import get_device
from rls.utils.sundry_utils import create_logger


class Base:

    def __init__(self, *args, **kwargs):
        '''
        inputs:
            a_dim: action spaces
            is_continuous: action type, refer to whether this control problem is continuous(True) or discrete(False)
            base_dir: the directory that store data, like model, logs, and other data
        '''
        super().__init__()
        base_dir = kwargs.get('base_dir')
        tf_dtype = str(kwargs.get('tf_dtype'))
        tf.keras.backend.set_floatx(tf_dtype)
        tf.random.set_seed(int(kwargs.get('seed', 0)))
        self.device = get_device()

        self.cp_dir, self.log_dir, self.excel_dir = [os.path.join(base_dir, i) for i in ['model', 'log', 'excel']]

        self.logger = create_logger(
            name='rls.algo.base',
            logger2file=bool(kwargs.get('logger2file', False)),
            file_name=self.log_dir + 'log.txt'
        )

        self.check_or_create(self.cp_dir, 'checkpoints')
        self.check_or_create(self.log_dir, 'logs(summaries)')
        if 1 == 0:  # Not used
            import pandas as pd
            self.check_or_create(self.excel_dir, 'excel')
            self.excel_writer = pd.ExcelWriter(self.excel_dir + '/data.xlsx')

        self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)  # in TF 2.x must be tf.int64, because function set_step need args to be tf.int64.
        self.cast = self._cast(dtype=tf_dtype)

    def _cast(self, dtype: str = 'float32') -> Callable:
        '''
        TODO: Annotation
        '''
        if dtype == 'float32':
            func = cast2float32
            self._data_type = tf.float32
        elif dtype == 'float64':
            self._data_type = tf.float64
            func = cast2float64
        else:
            raise Exception('Cast to this type has not been implemented.')

        def inner(*args, **kwargs):
            with tf.device(self.device):
                return func(*args, **kwargs)
        return inner

    def data_convert(self, data: Union[np.ndarray, List]) -> tf.Tensor:
        '''
        TODO: Annotation
        '''
        return tf.convert_to_tensor(data, dtype=self._data_type)

    def get_init_train_step(self) -> int:
        """
        get the initial training step. use for continue train from last training step.
        """
        if os.path.exists(os.path.join(self.cp_dir, 'checkpoint')):
            return int(tf.train.latest_checkpoint(self.cp_dir).split('-')[-1])
        else:
            return 0

    def get_init_training_info(self) -> Dict:
        '''
        TODO: Annotation
        '''
        path = f'{self.log_dir}/step.json'
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
        else:
            data = {}
        return dict(
            train_step=int(data.get('train_step', 0)),
            frame_step=int(data.get('frame_step', 0)),
            episode=int(data.get('episode', 0))
        )

    def _create_saver(self, kwargs: Dict) -> NoReturn:
        """
        create checkpoint and saver.
        """
        self.checkpoint = tf.train.Checkpoint(**kwargs)
        self.saver = tf.train.CheckpointManager(self.checkpoint, directory=self.cp_dir, max_to_keep=5, checkpoint_name='ckpt')

    def _create_writer(self, log_dir: str) -> tf.summary.SummaryWriter:
        self.check_or_create(log_dir, 'logs(summaries)')
        return tf.summary.create_file_writer(log_dir)

    def init_or_restore(self, base_dir: str) -> NoReturn:
        """
        check whether chekpoint and model be within cp_dir, if in it, restore otherwise initialize randomly.
        """
        cp_dir = os.path.join(base_dir, 'model')
        if os.path.exists(os.path.join(cp_dir, 'checkpoint')):
            try:
                self.checkpoint.restore(tf.train.latest_checkpoint(cp_dir)).expect_partial()    # 从指定路径导入模型
            except:
                self.logger.error(f'restore model from {cp_dir} FAILED.')
                raise Exception(f'restore model from {cp_dir} FAILED.')
            else:
                self.logger.info(f'restore model from {cp_dir} SUCCUESS.')
        else:
            self.checkpoint.restore(self.saver.latest_checkpoint).expect_partial()  # 从本模型目录载入模型，断点续训
            self.logger.info(f'restore model from {self.saver.latest_checkpoint} SUCCUESS.')
            self.logger.info('initialize model SUCCUESS.')

    def save_checkpoint(self, **kwargs) -> NoReturn:
        """
        save the training model 
        """
        train_step = int(kwargs.get('train_step', 0))
        self.saver.save(checkpoint_number=train_step)
        self.logger.info(f'Save checkpoint success. Training step: {train_step}')
        self.write_training_info(kwargs)

    def write_training_info(self, data: Dict) -> NoReturn:
        with open(f'{self.log_dir}/step.json', 'w') as f:
            json.dump(data, f)

    def writer_summary(self,
                       global_step: Union[int, tf.Variable],
                       writer: Optional[tf.summary.SummaryWriter] = None,
                       **kargs) -> NoReturn:
        """
        record the data used to show in the tensorboard
        """
        writer = writer or self.writer
        writer.set_as_default()
        tf.summary.experimental.set_step(global_step)
        for i in [{'tag': 'AGENT/' + key, 'value': kargs[key]} for key in kargs]:
            tf.summary.scalar(i['tag'], i['value'])
        writer.flush()

    def write_training_summaries(self,
                                 global_step: Union[int, tf.Variable],
                                 summaries: Dict,
                                 writer: Optional[tf.summary.SummaryWriter] = None) -> NoReturn:
        '''
        write tf summaries showing in tensorboard.
        '''
        writer = writer or self.writer
        writer.set_as_default()
        tf.summary.experimental.set_step(global_step)
        for key, value in summaries.items():
            tf.summary.scalar(key, value)
        writer.flush()

    def check_or_create(self, dicpath: str, name: str = '') -> NoReturn:
        """
        check dictionary whether existing, if not then create it.
        """
        if not os.path.exists(dicpath):
            os.makedirs(dicpath)
            self.logger.info(''.join([f'create {name} directionary :', dicpath]))

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
        self.global_step.assign(num)

    def show_logo(self) -> NoReturn:
        raise NotImplementedError
