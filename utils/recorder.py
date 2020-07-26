import os
import sys
import json
import logging
import pandas as pd
import tensorflow as tf

from typing import Dict


class Recorder(object):
    '''
    TF 2.0 Recorder
    '''

    def __init__(self, log_dir, excel_dir, logger2file):
        self.log_dir = log_dir
        self.excel_writer = pd.ExcelWriter(excel_dir + '/data.xlsx')
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

    def write_training_info(self, data: Dict) -> None:
        with open(f'{self.log_dir}/step.json', 'w') as f:
            json.dump(data, f)

    def get_training_info(self) -> Dict:
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