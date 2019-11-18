import os
import numpy as np
import tensorflow as tf
from abc import abstractmethod
from utils.recorder import RecorderTf2 as Recorder


class Base(tf.keras.Model):

    def __init__(self, a_dim_or_list, action_type, base_dir):
        '''
        inputs:
            a_dim_or_list: action spaces, if continuous, it will like [2,], if discrete, it will like [2,3,4]
            action_type: action type, refer to whether this control problem is 'continuous' or 'discrete'
            base_dir: the directory that store data, like model, logs, and other data
        '''
        super().__init__()
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            self.device = "/gpu:0"
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        else:
            self.device = "/cpu:0"
        tf.keras.backend.set_floatx('float32')
        self.cp_dir, self.log_dir, self.excel_dir = [os.path.join(base_dir, i) for i in ['model', 'log', 'excel']]
        self.action_type = action_type
        self.a_counts = int(np.array(a_dim_or_list).prod())
        self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)  # in TF 2.x must be tf.int64, because function set_step need args to be tf.int64.
        self.episode = 0    # episode of now
        self.IS_w = 1       # the weights of NN variables by using Importance sampling.

    @abstractmethod
    def choose_action(self, s, visual_s):
        '''
        choose actions while training.
        Input: 
            s: vector observation
            visual_s: visual observation
        Output: 
            actions
        '''
        pass
    
    @abstractmethod
    def choose_inference_action(self, s, visual_s):
        '''
        choose actions while inferencing.
        Input: 
            s: vector observation
            visual_s: visual observation
        Output: 
            actions
        '''
        pass

    def get_init_episode(self):
        """
        get the initial training step. use for continue train from last training step.
        """
        if os.path.exists(os.path.join(self.cp_dir, 'checkpoint')):
            return int(tf.train.latest_checkpoint(self.cp_dir).split('-')[-1])
        else:
            return 0

    def generate_recorder(self, logger2file, model=None):
        """
        create model/log/data dictionary and define writer to record training data.
        """

        self.check_or_create(self.cp_dir, 'checkpoints')
        self.check_or_create(self.log_dir, 'logs(summaries)')
        self.check_or_create(self.excel_dir, 'excel')
        self.recorder = Recorder(
            cp_dir=self.cp_dir,
            log_dir=self.log_dir,
            excel_dir=self.excel_dir,
            logger2file=logger2file,
            model=model
        )

    def init_or_restore(self, base_dir):
        """
        check whether chekpoint and model be within cp_dir, if in it, restore otherwise initialize randomly.
        """
        cp_dir = os.path.join(base_dir, 'model')
        if os.path.exists(os.path.join(cp_dir, 'checkpoint')):
            try:
                self.recorder.checkpoint.restore(self.recorder.saver.latest_checkpoint)
            except:
                self.recorder.logger.error('restore model from checkpoint FAILED.')
            else:
                self.recorder.logger.info('restore model from checkpoint SUCCUESS.')
        else:
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
        for i in [{'tag': 'MAIN/' + key, 'value': kargs[key]} for key in kargs]:
            tf.summary.scalar(i['tag'], i['value'])
        self.recorder.writer.flush()

    def check_or_create(self, dicpath, name=''):
        """
        check dictionary whether existing, if not then create it.
        """
        if not os.path.exists(dicpath):
            os.makedirs(dicpath)
            print(f'create {name} directionary :', dicpath)

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
        self.global_step = num

    def update_target_net_weights(self, tge, src, ployak=None):
        '''
        update weights of target neural network.
        '''
        if ployak is None:
            tf.group([t.assign(s) for t, s in zip(tge, src)])
        else:
            tf.group([t.assign(self.ployak * t + (1 - self.ployak) * s) for t, s in zip(tge, src)])
    

    def write_training_summaries(self, summaries:dict):
        '''
        write tf summaries showing in tensorboard.
        '''
        for key, value in summaries.items():
            tf.summary.scalar(key, value)