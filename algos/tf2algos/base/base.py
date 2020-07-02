import os
import logging
import numpy as np
import tensorflow as tf
from utils.tf2_utils import cast2float32, cast2float64
from utils.tf2_utils import get_device
from utils.recorder import Recorder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("algos.base")

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
        tf.random.set_seed(int(kwargs.get('seed', 0)))
        self.device = get_device()
        self.logger2file = bool(kwargs.get('logger2file', False))

        tf.keras.backend.set_floatx(tf_dtype)
        self.cp_dir, self.log_dir, self.excel_dir = [os.path.join(base_dir, i) for i in ['model', 'log', 'excel']]
        self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)  # in TF 2.x must be tf.int64, because function set_step need args to be tf.int64.
        self.cast = self._cast(dtype=tf_dtype)

    def _cast(self, dtype='float32'):
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
    
    def data_convert(self, data):
        '''
        TODO: Annotation
        '''
        return tf.convert_to_tensor(data, dtype=self._data_type)

    def get_init_episode(self):
        """
        get the initial training step. use for continue train from last training step.
        """
        if os.path.exists(os.path.join(self.cp_dir, 'checkpoint')):
            return int(tf.train.latest_checkpoint(self.cp_dir).split('-')[-1])
        else:
            return 0

    def generate_recorder(self, kwargs):
        """
        create model/log/data dictionary and define writer to record training data.
        """

        self.check_or_create(self.cp_dir, 'checkpoints')
        self.check_or_create(self.log_dir, 'logs(summaries)')
        self.check_or_create(self.excel_dir, 'excel')
        self.recorder = Recorder(
            kwargs,
            cp_dir=self.cp_dir,
            log_dir=self.log_dir,
            excel_dir=self.excel_dir,
            logger2file=self.logger2file
        )

    def init_or_restore(self, base_dir):
        """
        check whether chekpoint and model be within cp_dir, if in it, restore otherwise initialize randomly.
        """
        cp_dir = os.path.join(base_dir, 'model')
        if os.path.exists(os.path.join(cp_dir, 'checkpoint')):
            try:
                self.recorder.checkpoint.restore(tf.train.latest_checkpoint(cp_dir)).expect_partial()    # 从指定路径导入模型
            except:
                self.recorder.logger.error(f'restore model from {cp_dir} FAILED.')
                raise Exception(f'restore model from {cp_dir} FAILED.')
            else:
                self.recorder.logger.info(f'restore model from {cp_dir} SUCCUESS.')
        else:
            self.recorder.checkpoint.restore(self.recorder.saver.latest_checkpoint).expect_partial()  # 从本模型目录载入模型，断点续训
            self.recorder.logger.info(f'restore model from {self.recorder.saver.latest_checkpoint} SUCCUESS.')
            self.recorder.logger.info('initialize model SUCCUESS.')

    def save_checkpoint(self, global_step):
        """
        save the training model 
        """
        self.recorder.logger.info(f'Save checkpoint success. Episode: {global_step}')
        self.recorder.saver.save(checkpoint_number=global_step)

    def writer_summary(self, global_step, **kargs):
        """
        record the data used to show in the tensorboard
        """
        self.recorder.writer.set_as_default()
        tf.summary.experimental.set_step(global_step)
        for i in [{'tag': 'AGENT/' + key, 'value': kargs[key]} for key in kargs]:
            tf.summary.scalar(i['tag'], i['value'])
        self.recorder.writer.flush()

    def write_training_summaries(self, global_step, summaries: dict):
        '''
        write tf summaries showing in tensorboard.
        '''
        self.recorder.writer.set_as_default()
        tf.summary.experimental.set_step(global_step)
        for key, value in summaries.items():
            tf.summary.scalar(key, value)
        self.recorder.writer.flush()

    def check_or_create(self, dicpath, name=''):
        """
        check dictionary whether existing, if not then create it.
        """
        if not os.path.exists(dicpath):
            os.makedirs(dicpath)
            logger.info(''.join([f'create {name} directionary :', dicpath]))

    def save_weights(self, path):
        """
        save trained weights
        :return: None
        """
        # self.net.save_weights(os.path.join(path, 'net.ckpt'))
        pass

    def load_weights(self, path):
        """
        load trained weights
        :return: None
        """
        # self.net.load_weights(os.path.join(path, 'net.ckpt'))
        pass

    def close(self):
        """
        end training, and export the training model
        """
        pass

    def get_global_step(self):
        """
        get the current trianing step.
        """
        return self.global_step

    def set_global_step(self, num):
        """
        set the start training step.
        """
        self.global_step.assign(num)

    def show_logo(self):
        raise NotImplementedError
