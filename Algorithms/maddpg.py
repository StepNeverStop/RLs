import os
import sys
sys.path.append('..')
import numpy as np
import tensorflow as tf
import Nn
from utils.recorder import Recorder
from tensorflow.python.tools import freeze_graph
from mlagents.trainers import tensorflow_to_barracuda as tf2bc


class MADDPG(object):
    _version_number_ = 2

    def __init__(self,
                 s_dim,
                 a_dim_or_list,
                 action_type,
                 gamma=0.99,
                 ployak=0.995,
                 lr=5.0e-4,
                 max_episode=50000,
                 batch_size=100,
                 buffer_size=10000,
                 n=1,
                 i=0,
                 cp_dir=None,
                 log_dir=None,
                 excel_dir=None,
                 logger2file=False,
                 out_graph=False):
        assert action_type == 'continuous'

        self.graph = tf.Graph()
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options), graph=self.graph)
        self.n= n
        self.i = i
        self.s_dim = s_dim
        self.a_dim_or_list = a_dim_or_list
        self.action_type = action_type
        self.a_counts = np.array(a_dim_or_list).prod()
        self.gamma = gamma
        self.max_episode = max_episode
        self.cp_dir = cp_dir
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.ployak = ployak
        self.possible_output_nodes = ['action', 'version_number', 'is_continuous_control', 'action_output_shape', 'memory_size']
        self.init_step = self.get_init_step()
        
        with self.graph.as_default():
            tf.set_random_seed(-1)  # variables initialization consistent.
            tf.Variable(1 if action_type == 'continuous' else 0, name='is_continuous_control', trainable=False, dtype=tf.int32)  # continuous 1 discrete 0
            tf.Variable(self.a_counts, name="action_output_shape", trainable=False, dtype=tf.int32)
            tf.Variable(self._version_number_, name='version_number', trainable=False, dtype=tf.int32)
            tf.Variable(0, name="memory_size", trainable=False, dtype=tf.int32)

            self.q_actor_s = tf.placeholder(tf.float32, [None, (self.s_dim)*self.n], 'q_actor_s')
            self.q_actor_a_previous = tf.placeholder(tf.float32, [None, (self.a_counts)*self.i], 'q_actor_a_previous')
            self.q_actor_a_later = tf.placeholder(tf.float32, [None, (self.a_counts)*(self.n-self.i-1)], 'q_actor_a_later')
            self.q_input = tf.placeholder(tf.float32, [None,(self.s_dim+self.a_counts)*self.n], 'q_input')
            self.q_target_input = tf.placeholder(tf.float32, [None,(self.s_dim+self.a_counts)*self.n], 'q_target_input')
            self.pl_s = tf.placeholder(tf.float32, [None, self.s_dim], 'vector_observation')
            self.pl_r = tf.placeholder(tf.float32, [None, 1], 'reward')
            self.episode = tf.Variable(tf.constant(0))
            self.global_step = tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer(value=self.init_step), trainable=False)

            self.lr = tf.train.polynomial_decay(lr, self.episode, self.max_episode, 1e-10, power=1.0)
            # self.action_noise = Nn.NormalActionNoise(mu=np.zeros(self.a_counts), sigma=1 * np.ones(self.a_counts))
            self.action_noise = Nn.OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.a_counts), sigma=0.2 * np.ones(self.a_counts))
            self.mu = Nn.actor_dpg('actor', self.pl_s, self.a_counts, True)
            tf.identity(self.mu, 'action')
            self.action = tf.clip_by_value(self.mu + self.action_noise(), -1, 1)

            self.target_mu = Nn.actor_dpg('actor_target', self.pl_s, self.a_counts, False)
            self.action_target = tf.clip_by_value(self.target_mu + self.action_noise(), -1, 1)

            self.ss_mu = tf.concat((self.q_actor_s, self.q_actor_a_previous, self.mu, self.q_actor_a_later), axis=1)

            self.q = Nn.critic_q_one('q', self.q_input, True, reuse=False)
            self.q_actor = Nn.critic_q_one('q', self.ss_mu, True, reuse=True)
            self.q_target = Nn.critic_q_one('q_target', self.q_target_input, False, reuse=False)
            self.dc_r = tf.stop_gradient(self.pl_r + self.gamma * self.q_target)

            self.q_loss = 0.5 * tf.reduce_mean(tf.squared_difference(self.q, self.dc_r))
            self.actor_loss = -tf.reduce_mean(self.q_actor)

            self.q_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='q')
            self.q_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_target')
            self.actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
            self.actor_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_target')

            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_q = optimizer.minimize(
                self.q_loss, var_list=self.q_vars, global_step=self.global_step)
            with tf.control_dependencies([self.train_q]):
                self.train_actor = optimizer.minimize(self.actor_loss, var_list=self.actor_vars)
            with tf.control_dependencies([self.train_actor]):
                self.assign_q_target = tf.group([tf.assign(r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.q_target_vars, self.q_vars)])
                self.assign_actor_target = tf.group([tf.assign(r, self.ployak * v + (1 - self.ployak) * r) for r, v in zip(self.actor_target_vars, self.actor_vars)])
            self.train_sequence = [self.train_q, self.train_actor, self.assign_q_target, self.assign_actor_target]

            tf.summary.scalar('LOSS/actor_loss', tf.reduce_mean(self.actor_loss))
            tf.summary.scalar('LOSS/critic_loss', tf.reduce_mean(self.q_loss))
            tf.summary.scalar('LEARNING_RATE/lr', tf.reduce_mean(self.lr))
            self.summaries = tf.summary.merge_all()
            self.generate_recorder(
                cp_dir=cp_dir,
                log_dir=log_dir,
                excel_dir=excel_dir,
                logger2file=logger2file,
                graph=self.graph if out_graph else None
            )
            self.recorder.logger.info('''
　　ｘｘｘｘ　　　　ｘｘｘ　　　　　　　　　ｘｘ　　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　ｘｘｘ　　　　ｘｘ　　　　　　　　　ｘｘｘ　　　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘｘ　　ｘｘ　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　　　ｘｘｘ　　ｘｘｘ　　　　　　　　　ｘｘｘ　　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　ｘｘｘ　　ｘｘｘ　　　　　　　　　ｘ　ｘｘ　　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　ｘｘ　　　　　　　　　　
　　　　ｘｘｘｘ　ｘ　ｘ　　　　　　　　ｘｘ　ｘｘ　　　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘ　　　ｘｘｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　　　ｘ　　　ｘｘｘｘｘ　　　
　　　　ｘ　ｘｘｘｘ　ｘ　　　　　　　　ｘｘｘｘｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　ｘｘｘ　　　　
　　　　ｘ　ｘｘｘ　　ｘ　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　ｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘ　　　　ｘ　　　　　
　　　　ｘ　　ｘｘ　　ｘ　　　　　　　ｘｘ　　　ｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　ｘｘｘ　　　　　　　　　ｘ　　　　　　　　　　　　ｘｘｘ　　ｘｘ　　　　　
　　ｘｘｘｘ　ｘｘｘｘｘｘ　　　　　ｘｘｘ　　ｘｘｘｘｘ　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘｘｘ　　　　　　　　ｘｘｘｘｘ　　　　　　　　　　　ｘｘｘｘｘｘ　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ｘｘ　　　
            ''')
            self.recorder.logger.info(self.action_noise)
            self.init_or_restore(cp_dir)

    def choose_action(self, s):
        return self.sess.run(self.action, feed_dict={
            self.pl_s: s,
        })
    
    def get_target_action(self, s):
        return self.sess.run(self.action_target, feed_dict={
            self.pl_s: s,
        })

    def choose_inference_action(self, s):
        return self.sess.run(self.mu, feed_dict={
            self.pl_s: s,
        })

    def learn(self, episode, ss, ap, al, s_a, s_a_, s, r):
        summaries, _ = self.sess.run([self.summaries, self.train_sequence], feed_dict={
            self.q_actor_s: ss,
            self.q_actor_a_previous: ap,
            self.q_actor_a_later: al,
            self.q_input: s_a,
            self.q_target_input: s_a_,
            self.pl_s: s,
            self.pl_r: r,
            self.episode: episode
        })
        self.recorder.writer.add_summary(summaries, self.sess.run(self.global_step))

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