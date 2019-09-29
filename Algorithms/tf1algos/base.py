import os
import numpy as np
import tensorflow as tf
from utils.recorder import Recorder
from tensorflow.python.tools import freeze_graph
from mlagents.trainers import tensorflow_to_barracuda as tf2bc


class Base(object):
    _version_number_ = 2

    def __init__(self, a_dim_or_list, action_type, base_dir):

        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=self.graph)
        self.cp_dir, self.log_dir, self.excel_dir = [os.path.join(base_dir, i) for i in ['model', 'log', 'excel']]
        self.action_type = action_type
        self.a_counts = int(np.array(a_dim_or_list).prod())

        self.possible_output_nodes = ['action', 'version_number', 'is_continuous_control', 'action_output_shape', 'memory_size']

        with self.graph.as_default():
            tf.set_random_seed(-1)  # variables initialization consistent.
            tf.Variable(1 if action_type == 'continuous' else 0, name='is_continuous_control', trainable=False, dtype=tf.int32)  # continuous 1 discrete 0
            tf.Variable(self.a_counts, name="action_output_shape", trainable=False, dtype=tf.int32)
            tf.Variable(self._version_number_, name='version_number', trainable=False, dtype=tf.int32)
            tf.Variable(0, name="memory_size", trainable=False, dtype=tf.int32)
            self.episode = tf.Variable(tf.constant(0))
            self.global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)

    def get_init_episode(self):
        """
        get the initial training step. use for continue train from last training step.
        """
        if os.path.exists(os.path.join(self.cp_dir, 'checkpoint')):
            return int(tf.train.latest_checkpoint(self.cp_dir).split('-')[-1])
        else:
            return 0

    def generate_recorder(self, logger2file, graph):
        """
        create model/log/data dictionary and define writer to record training data.
        """

        self.check_or_create(self.cp_dir, 'checkpoints')
        self.check_or_create(self.log_dir, 'logs(summaries)')
        self.check_or_create(self.excel_dir, 'excel')
        self.recorder = Recorder(
            log_dir=self.log_dir,
            excel_dir=self.excel_dir,
            logger2file=logger2file,
            graph=graph
        )

    def init_or_restore(self, base_dir):
        """
        check whether chekpoint and model be within cp_dir, if in it, restore otherwise initialize randomly.
        """
        cp_dir = os.path.join(base_dir, 'model')
        with self.graph.as_default():
            if os.path.exists(os.path.join(cp_dir, 'checkpoint')):
                try:
                    self.recorder.saver.restore(self.sess, tf.train.latest_checkpoint(cp_dir))
                except:
                    self.recorder.logger.error('restore model from checkpoint FAILED.')
                else:
                    self.recorder.logger.info('restore model from checkpoint SUCCUESS.')
            else:
                self.sess.run(tf.global_variables_initializer())
                self.recorder.logger.info('initialize model SUCCUESS.')

    def save_checkpoint(self, global_step):
        """
        save the training model 
        """
        self.recorder.saver.save(self.sess, os.path.join(self.cp_dir, 'rb'), global_step=global_step, write_meta_graph=False)

    def writer_summary(self, global_step, **kargs):
        """
        record the data used to show in the tensorboard
        """
        self.recorder.writer_summary(
            x=global_step,
            ys=[{'tag': 'MAIN/' + key, 'value': kargs[key]} for key in kargs]
        )

    def _process_graph(self):
        """
        Gets the list of the output nodes present in the graph for inference
        :return: list of node names
        """
        all_nodes = [x.name for x in self.graph.as_graph_def().node]
        nodes = [x for x in all_nodes if x in self.possible_output_nodes]
        return nodes

    def export_model(self):
        """
        Exports latest saved model to .nn format for Unity embedding.
        """
        tf.train.write_graph(self.graph, self.cp_dir, 'raw_graph_def.pb', as_text=False)
        with self.graph.as_default():
            target_nodes = ','.join(self._process_graph())
            freeze_graph.freeze_graph(
                input_graph=os.path.join(self.cp_dir, 'raw_graph_def.pb'),
                input_binary=True,
                input_checkpoint=tf.train.latest_checkpoint(self.cp_dir),
                output_node_names=target_nodes,
                output_graph=(os.path.join(self.cp_dir, 'frozen_graph_def.pb')),
                # output_graph=(os.path.join(self.cp_dir,'model.bytes')),
                clear_devices=True, initializer_nodes='', input_saver='',
                restore_op_name='save/restore_all',
                filename_tensor_name='save/Const:0')
        tf2bc.convert(os.path.join(self.cp_dir, 'frozen_graph_def.pb'), os.path.join(self.cp_dir, 'model.nn'))

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
        self.export_model()

    def get_global_step(self):
        """
        get the current trianing step.
        """
        return self.sess.run(self.global_step)

    def set_global_step(self, num):
        """
        set the start training step.
        """
        with self.graph.as_default():
            self.global_step.load(num, self.sess)
