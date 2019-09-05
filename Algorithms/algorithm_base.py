import os
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import tensorflow as tf
import Nn
from utils.recorder import Recorder
from utils.replay_buffer import ExperienceReplay, NStepExperienceReplay, PrioritizedExperienceReplay, NStepPrioritizedExperienceReplay
from tensorflow.python.tools import freeze_graph
from mlagents.trainers import tensorflow_to_barracuda as tf2bc


class Policy(object):
    _version_number_ = 2

    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolutions,
                 a_dim_or_list,
                 action_type,
                 gamma,
                 max_episode,
                 cp_dir,
                 policy_mode=None,
                 batch_size=1,
                 buffer_size=1,
                 use_priority=False,
                 n_step=False,
                 ):
        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=self.graph)
        self.s_dim = s_dim
        self.visual_sources = visual_sources
        self.visual_dim = [
            visual_sources,
            visual_resolutions[0]['height'],
            visual_resolutions[0]['width'],
            1 if visual_resolutions[0]['blackAndWhite'] else 3
        ] if visual_sources else [0]
        self.a_dim_or_list = a_dim_or_list
        self.action_type = action_type
        self.a_counts = np.array(a_dim_or_list).prod()
        self.gamma = gamma
        self.max_episode = max_episode
        self.cp_dir = cp_dir
        self.policy_mode = policy_mode
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.possible_output_nodes = ['action', 'version_number', 'is_continuous_control', 'action_output_shape', 'memory_size']
        self.init_step = self.get_init_step()

        '''
        the biggest diffenernce between policy_modes(ON and OFF) is 'OFF' mode need raise the dimension
        of 'r' and 'done'.
        'ON' mode means program will call on_store function and use pandas dataframe to store data.
        'OFF' mode will call off_store function and use replay buffer to store data.
        '''
        if self.policy_mode == 'ON':
            self.data = pd.DataFrame(columns=['s', 'a', 'r', 's_', 'done'])
        elif self.policy_mode == 'OFF':
            if use_priority:
                if n_step:
                    print('N-Step PER')
                    self.data = NStepPrioritizedExperienceReplay(self.batch_size, self.buffer_size, max_episode=self.max_episode,
                                                                 gamma=self.gamma, alpha=0.6, beta=0.2, epsilon=0.01, agents_num=20, n=4)
                else:
                    print('PER')
                    self.data = PrioritizedExperienceReplay(self.batch_size, self.buffer_size, max_episode=self.max_episode, alpha=0.6, beta=0.2, epsilon=0.01)
            else:
                if n_step:
                    print('N-Step ER')
                    self.data = NStepExperienceReplay(self.batch_size, self.buffer_size, gamma=self.gamma, agents_num=20, n=4)
                else:
                    print('ER')
                    self.data = ExperienceReplay(self.batch_size, self.buffer_size)

        else:
            raise Exception('Please specific a mode of policy!')

        with self.graph.as_default():
            # continuous 1 discrete 0
            tf.Variable(1 if action_type == 'continuous' else 0, name='is_continuous_control', trainable=False, dtype=tf.int32)
            tf.Variable(self.a_counts, name="action_output_shape", trainable=False, dtype=tf.int32)
            tf.Variable(self._version_number_, name='version_number', trainable=False, dtype=tf.int32)
            tf.Variable(0, name="memory_size", trainable=False, dtype=tf.int32)
            self.pl_s = tf.placeholder(tf.float32, [None, self.s_dim], 'vector_observation')
            self.pl_a = tf.placeholder(tf.float32, [None, self.a_counts], 'pl_action')
            self.pl_r = tf.placeholder(tf.float32, [None, 1], 'reward')
            self.pl_s_ = tf.placeholder(tf.float32, [None, self.s_dim], 'next_state')
            self.pl_done = tf.placeholder(tf.float32, [None, 1], 'done')
            self.pl_visual_s = tf.placeholder(tf.float32, [None] + self.visual_dim, 'visual_observation_')
            self.pl_visual_s_ = tf.placeholder(tf.float32, [None] + self.visual_dim, 'next_visual_observation')
            self.episode = tf.Variable(tf.constant(0))
            self.global_step = tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer(value=self.init_step), trainable=False)
            if visual_sources:
                self.s = tf.concat((Nn.visual_nn('visual_net', self.pl_visual_s), self.pl_s), axis=1)
                self.s_ = tf.concat((Nn.visual_nn('visual_net', self.pl_visual_s_), self.pl_s_), axis=1)
                self.conv_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='visual_net')
            else:
                self.s = self.pl_s
                self.s_ = self.pl_s_
                self.conv_vars = []

    def on_store(self, s, visual_s, a, r, s_, visual_s_, done):
        """
        for on-policy training, use this function to store <s, a, r, s_, done> into DataFrame of Pandas.
        """
        assert isinstance(a, np.ndarray)
        assert isinstance(r, np.ndarray)
        assert isinstance(done, np.ndarray)
        self.data = self.data.append({
            's': s,
            'visual_s': visual_s,
            'a': a,
            'r': r,
            's_': s_,
            'visual_s_': visual_s_,
            'done': done
        }, ignore_index=True)

    def off_store(self, s, visual_s, a, r, s_, visual_s_, done):
        """
        for off-policy training, use this function to store <s, a, r, s_, done> into ReplayBuffer.
        """
        assert isinstance(a, np.ndarray)
        assert isinstance(r, np.ndarray)
        assert isinstance(done, np.ndarray)
        self.data.add(s, visual_s, a, r, s_, visual_s_, done)

    def no_op_store(self, s, visual_s, a, r, s_, visual_s_, done):
        assert isinstance(a, np.ndarray)
        assert isinstance(r, np.ndarray)
        assert isinstance(done, np.ndarray)
        if self.policy_mode == 'OFF':
            self.data.add(s, visual_s, a, r[:, np.newaxis], s_, visual_s_, done[:, np.newaxis])

    def clear(self):
        """
        clear the DataFrame.
        """
        self.data.drop(self.data.index, inplace=True)

    def get_init_step(self):
        """
        get the initial training step. use for continue train from last training step.
        """
        if os.path.exists(os.path.join(self.cp_dir, 'checkpoint')):
            return int(tf.train.latest_checkpoint(self.cp_dir).split('-')[-1])
        else:
            return 0

    def get_max_episode(self):
        """
        get the max episode of this training model.
        """
        return self.max_episode

    def generate_recorder(self, cp_dir, log_dir, excel_dir, logger2file, graph):
        """
        create model/log/data dictionary and define writer to record training data.
        """
        self.check_or_create(cp_dir, 'checkpoints')
        self.check_or_create(log_dir, 'logs(summaries)')
        self.check_or_create(excel_dir, 'excel')
        self.recorder = Recorder(
            log_dir=log_dir,
            excel_dir=excel_dir,
            logger2file=logger2file,
            graph=graph
        )

    def init_or_restore(self, cp_dir):
        """
        check whether chekpoint and model be within cp_dir, if in it, restore otherwise initialize randomly.
        """
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
