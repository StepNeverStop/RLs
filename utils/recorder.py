import sys
import logging
import tensorflow as tf
import pandas as pd


class Recorder(object):
    '''
    TF 1.x Recorder
    '''
    def __init__(self, log_dir, excel_dir, logger2file, graph=None):
        self.saver = tf.train.Saver(max_to_keep=5, pad_step_number=True)
        self.writer = tf.summary.FileWriter(log_dir, graph=graph)
        self.excel_writer = pd.ExcelWriter(excel_dir + 'data.xlsx')
        self.logger = self.create_logger(
            name='logger',
            console_level=logging.INFO,
            console_format='%(levelname)s : %(message)s',
            logger2file=logger2file,
            file_name=log_dir + 'log.txt',
            file_level=logging.WARNING,
            file_format='%(lineno)d - %(asctime)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s'
        )

    def create_logger(self, name, console_level, console_format, logger2file, file_name, file_level, file_format):
        logger = logging.Logger(name)
        logger.setLevel(level=console_level)
        stdout_handle = logging.StreamHandler(stream=sys.stdout)
        stdout_handle.setFormatter(logging.Formatter(console_format if console_level > 20 else '%(message)s'))
        logger.addHandler(stdout_handle)
        if logger2file:
            logfile_handle = logging.FileHandler(file_name)
            logfile_handle.setLevel(file_level)
            logfile_handle.setFormatter(logging.Formatter(file_format))
            logger.addHandler(logfile_handle)
        return logger

    def writer_summary(self, x, ys):
        if self.writer is not None:
            self.writer.add_summary(tf.Summary(
                value=[
                    tf.Summary.Value(tag=y['tag'], simple_value=y['value']) for y in ys
                ]), x)

class RecorderTf2(object):
    '''
    TF 2.0 Recorder
    '''
    def __init__(self, cp_dir, log_dir, excel_dir, logger2file, model=None):
        self.writer = tf.summary.create_file_writer(log_dir)
        self.writer.set_as_default()
        self.checkpoint = tf.train.Checkpoint(policy=model)
        self.saver = tf.train.CheckpointManager(self.checkpoint, directory=cp_dir, max_to_keep=5, checkpoint_name='rb')
        self.excel_writer = pd.ExcelWriter(excel_dir + 'data.xlsx')
        self.logger = self.create_logger(
            name='logger',
            console_level=logging.INFO,
            console_format='%(levelname)s : %(message)s',
            logger2file=logger2file,
            file_name=log_dir + 'log.txt',
            file_level=logging.WARNING,
            file_format='%(lineno)d - %(asctime)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s'
        )

    def create_logger(self, name, console_level, console_format, logger2file, file_name, file_level, file_format):
        logger = logging.Logger(name)
        logger.setLevel(level=console_level)
        stdout_handle = logging.StreamHandler(stream=sys.stdout)
        stdout_handle.setFormatter(logging.Formatter(console_format if console_level > 20 else '%(message)s'))
        logger.addHandler(stdout_handle)
        if logger2file:
            logfile_handle = logging.FileHandler(file_name)
            logfile_handle.setLevel(file_level)
            logfile_handle.setFormatter(logging.Formatter(file_format))
            logger.addHandler(logfile_handle)
        return logger
