import os
import sys
sys.path.append('..')
import pandas as pd
import tensorflow as tf
from utils.recorder import Recorder
from utils.replay_buffer import ReplayBuffer
from tensorflow.python.tools import freeze_graph


class Policy(object):
    _version_number_ = 2

    def __init__(self,
                 s_dim,
                 a_counts,
                 cp_dir,
                 policy_mode=None,
                 batch_size=1,
                 buffer_size=1,
                 use_priority=False
                 ):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options),
                               graph=self.graph)
        self.s_dim = s_dim
        self.a_counts = a_counts
        self.cp_dir = cp_dir
        self.activation_fn = tf.nn.tanh
        self.policy_mode = policy_mode
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.possible_output_nodes = ['action', 'version_number']
        if self.policy_mode == 'ON':
            self.data = pd.DataFrame(columns=['s', 'a', 'r', 's_'])
        elif self.policy_mode == 'OFF':
            self.data = ReplayBuffer(self.batch_size, self.buffer_size)
        else:
            raise Exception('Please specific a mode of policy!')

        with self.graph.as_default():
            tf.Variable(self._version_number_, name='version_number',
                        trainable=False, dtype=tf.int32)
            self.pl_s = tf.placeholder(tf.float32, [None, self.s_dim], 'pl_state')
            self.pl_a = tf.placeholder(
                tf.float32, [None, self.a_counts], 'pl_action')
            self.init_step = self.get_init_step(
                cp_dir=cp_dir)
            self.global_step = tf.get_variable('global_step', shape=(
            ), initializer=tf.constant_initializer(value=self.init_step), trainable=False)

    def on_store(self, s, a, r, s_, done):
        self.data = self.data.append({
            's': s,
            'a': a,
            'r': r,
            's_': s_,
            'done': done
        }, ignore_index=True)

    def off_store(self, s, a, r, s_, done):
        self.data.add(s, a, r, s_, done)

    def clear(self):
        self.data.drop(self.data.index, inplace=True)

    def get_init_step(self, cp_dir):
        if os.path.exists(cp_dir + 'checkpoint'):
            return int(tf.train.latest_checkpoint(cp_dir).split('-')[-1])
        else:
            return 0

    def generate_recorder(self, cp_dir, log_dir, excel_dir, logger2file, graph):
        self.check_or_create(cp_dir, 'checkpoints')
        self.check_or_create(log_dir, 'logs(summaries)')
        self.check_or_create(excel_dir, 'excel')
        self.recorder = Recorder(
            log_dir=log_dir,
            excel_dir=excel_dir,
            logger2file=logger2file,
            graph=graph
        )

    def init_or_restore(self, cp_dir, sess):
        if os.path.exists(cp_dir + 'checkpoint'):
            try:
                self.recorder.saver.restore(
                    sess, tf.train.latest_checkpoint(cp_dir))
                self.sess.run(self.global_step.initializer)
            except:
                self.recorder.logger.error(
                    'restore model from checkpoint FAILED.')
            else:
                self.recorder.logger.info(
                    'restore model from checkpoint SUCCUESS.')
        else:
            sess.run(tf.global_variables_initializer())
            self.recorder.logger.info('initialize model SUCCUESS.')

    def save_checkpoint(self, global_step):
        self.recorder.saver.save(
            self.sess, self.cp_dir + 'rb', global_step=global_step, write_meta_graph=False)

    def writer_summary(self, global_step, **kargs):
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
        print(nodes)
        return nodes

    def export_model(self):
        """
        Exports latest saved model to .nn format for Unity embedding.
        """
        tf.train.write_graph(
            self.graph, self.cp_dir, 'raw_graph_def.pb', as_text=False)
        with self.graph.as_default():
            target_nodes = ','.join(self._process_graph())
            freeze_graph.freeze_graph(
                input_graph=self.cp_dir + 'raw_graph_def.pb',
                input_binary=True,
                input_checkpoint=tf.train.latest_checkpoint(self.cp_dir),
                output_node_names=target_nodes,
                output_graph=(self.cp_dir + 'frozen_graph_def.pb'),
                # output_graph=(self.cp_dir + '/model.bytes'),
                clear_devices=True, initializer_nodes='', input_saver='',
                restore_op_name='save/restore_all',
                filename_tensor_name='save/Const:0')
        # tf2bc.convert(self.cp_dir + 'frozen_graph_def.pb',
        #               self.cp_dir + '.nn')

    def check_or_create(self, dicpath, name=''):
        if not os.path.exists(dicpath):
            os.makedirs(dicpath)
            print(f'create {name} directionary :', dicpath)

    def close(self):
        self.export_model()
