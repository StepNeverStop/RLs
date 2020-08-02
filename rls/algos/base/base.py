#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
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

from rls.utils.tf2_utils import get_device
from rls.utils.display import colorize
from rls.utils.sundry_utils import \
    check_or_create
from rls.utils.logging_utils import \
    get_logger, \
    set_log_file
logger = get_logger(__name__)


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
        self._tf_data_type = tf.float32 if tf_dtype == 'float32' else tf.float64
        tf.keras.backend.set_floatx(tf_dtype)

        tf.random.set_seed(int(kwargs.get('seed', 0)))
        self.device = get_device()

        self.cp_dir, self.log_dir, self.excel_dir = [os.path.join(base_dir, i) for i in ['model', 'log', 'excel']]

        check_or_create(self.cp_dir, 'checkpoints(models)')
        check_or_create(self.log_dir, 'logs(summaries)')
        if bool(kwargs.get('logger2file', False)):
            set_log_file(log_file=os.path.join(self.log_dir, 'log.txt'))

        if 1 == 0:  # Not used
            import pandas as pd
            check_or_create(self.excel_dir, 'excel')
            self.excel_writer = pd.ExcelWriter(self.excel_dir + '/data.xlsx')

        self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)  # in TF 2.x must be tf.int64, because function set_step need args to be tf.int64.
        self._worker_params_dict = {}
        self._residual_params_dict = dict(global_step=self.global_step)

    def _tf_data_cast(self, *args):
        '''
        TODO: Annotation
        '''
        with tf.device(self.device):
            return [tf.cast(i, self._tf_data_type) for i in args]

    def data_convert(self, data: Union[np.ndarray, List]) -> tf.Tensor:
        '''
        TODO: Annotation
        '''
        with tf.device(self.device):
            return tf.convert_to_tensor(data, dtype=self._tf_data_type)

    def get_init_train_step(self) -> int:
        """
        get the initial training step. use for continue train from last training step.
        """
        if os.path.exists(os.path.join(self.cp_dir, 'checkpoint')):
            return int(tf.train.latest_checkpoint(self.cp_dir).split('-')[-1])
        else:
            return 0

    def _create_saver(self) -> NoReturn:
        """
        create checkpoint and saver.
        """
        self.checkpoint = tf.train.Checkpoint(**self._worker_params_dict, **self._residual_params_dict)
        self.saver = tf.train.CheckpointManager(self.checkpoint, directory=self.cp_dir, max_to_keep=5, checkpoint_name='ckpt')

    def _create_writer(self, log_dir: str) -> tf.summary.SummaryWriter:
        check_or_create(log_dir, 'logs(summaries)')
        return tf.summary.create_file_writer(log_dir)

    def init_or_restore(self, base_dir: Optional[str] = None) -> NoReturn:
        """
        check whether chekpoint and model be within cp_dir, if in it, restore otherwise initialize randomly.
        """
        if base_dir is not None:
            cp_dir = os.path.join(base_dir, 'model')
            if os.path.exists(os.path.join(cp_dir, 'checkpoint')):
                try:
                    ckpt = tf.train.latest_checkpoint(cp_dir)
                    self.checkpoint.restore(ckpt).expect_partial()    # 从指定路径导入模型
                except:
                    logger.error(colorize(f'restore model from {cp_dir} FAILED.', color='red'))
                    raise Exception(f'restore model from {cp_dir} FAILED.')
                else:
                    logger.info(colorize(f'restore model from {ckpt} SUCCUESS.', color='green'))
        else:
            ckpt = self.saver.latest_checkpoint
            self.checkpoint.restore(ckpt).expect_partial()  # 从本模型目录载入模型，断点续训
            logger.info(colorize(f'restore model from {ckpt} SUCCUESS.', color='green'))
            logger.info(colorize('initialize model SUCCUESS.', color='green'))

    def save_checkpoint(self, **kwargs) -> NoReturn:
        """
        save the training model 
        """
        train_step = int(kwargs.get('train_step', 0))
        self.saver.save(checkpoint_number=train_step)
        logger.info(f'Save checkpoint success. Training step: {train_step}')
        self.write_training_info(kwargs)

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

    def write_training_info(self, data: Dict) -> NoReturn:
        '''
        TODO: Annotation
        '''
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

    def get_worker_params(self):
        weights_list =  list(map(lambda x: x.numpy(), self._worker_params_list))
        logger.info('get worker params success.')
        return weights_list

    def set_worker_params(self, weights_list):
        [src.assign(tgt) for src, tgt in zip(self._worker_params_list, weights_list)]
        logger.info('set worker params success.')

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

    def _model_post_process(self) -> NoReturn:
        self._worker_params_list = []
        for k, v in self._worker_params_dict.items():
            if isinstance(v, tf.keras.Model):
                self._worker_params_list.extend(list(map(lambda x: x, v.weights)))
            else:
                self._worker_params_list.append(v)
        self._create_saver()
