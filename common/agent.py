import os
import sys
import time
import logging
import numpy as np

from copy import deepcopy
from common.config import Config
from common.make_env import make_env
from common.yaml_ops import save_config, load_config
from common.train.gym import gym_train, gym_no_op, gym_inference
from common.train.unity import unity_train, unity_no_op, unity_inference
from common.train.unity import ma_unity_no_op, ma_unity_train, ma_unity_inference
from algos.register import get_model_info
from utils.replay_buffer import ExperienceReplay
from utils.time import get_time_hhmmss


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("common.agent")


def ShowConfig(config):
    for key in config:
        logger.info('-' * 60)
        logger.info(
            ''.join(['|', str(key).ljust(28), str(config[key]).rjust(28), '|'])
        )
    logger.info('-' * 60)


def UpdateConfig(config, file_path, key_name='algo'):
    _config = load_config(file_path)
    key_values = _config[key_name]
    try:
        for key in key_values:
            config[key] = key_values[key]
    except Exception as e:
        logger.info(e)
        sys.exit()
    return config


def get_buffer(buffer_args: Config):

    if buffer_args.get('buffer_size', 0) <= 0:
        logger.info('This algorithm does not need sepecify a data buffer oustside the model.')
        return None

    _type = buffer_args.get('type', None)

    if _type == 'ER':
        logger.info('ER')
        from utils.replay_buffer import ExperienceReplay as Buffer
    elif _type == 'PER':
        logger.info('PER')
        from utils.replay_buffer import PrioritizedExperienceReplay as Buffer
    elif _type == 'NstepER':
        logger.info('NstepER')
        from utils.replay_buffer import NStepExperienceReplay as Buffer
    elif _type == 'NstepPER':
        logger.info('NstepPER')
        from utils.replay_buffer import NStepPrioritizedExperienceReplay as Buffer
    elif _type == 'EpisodeER':
        logger.info('EpisodeER')
        from utils.replay_buffer import EpisodeExperienceReplay as Buffer
    else:
        logger.info('On-Policy DataBuffer')
        return None

    return Buffer(batch_size=buffer_args['batch_size'], capacity=buffer_args['buffer_size'], **buffer_args[_type].to_dict)


class Agent:
    def __init__(self, env_args: Config, model_args: Config, buffer_args: Config, train_args: Config):
        self.env_args = env_args
        self.model_args = model_args
        self.buffer_args = buffer_args
        self.train_args = train_args

        self.model_index = str(self.train_args.get('index'))
        self.start_time = time.time()
        self.all_learner_print = bool(self.train_args.get('all_learner_print', False))
        if '-' not in self.train_args['name']:
            self.train_args['name'] += f'-{self.model_index}'
        if self.model_args['load'] is None:
            self.train_args['load_model_path'] = os.path.join(self.train_args['base_dir'], self.train_args['name'])
        else:
            if '/' in self.model_args['load'] or '\\' in self.model_args['load']:   # 所有训练进程都以该模型路径初始化，绝对路径
                self.train_args['load_model_path'] = self.model_args['load']
            elif '-' in self.model_args['load']:
                self.train_args['load_model_path'] = os.path.join(self.train_args['base_dir'], self.model_args['load'])  # 指定了名称和序号，所有训练进程都以该模型路径初始化，相对路径
            else:   # 只写load的训练名称，不用带进程序号，会自动补
                self.train_args['load_model_path'] = os.path.join(self.train_args['base_dir'], self.model_args['load'] + f'-{self.model_index}')

        # ENV
        logger.info('Initialize environment begin...')
        self.env = make_env(self.env_args.to_dict)
        logger.info('Initialize environment successful.')

        # ALGORITHM CONFIG
        Model, algorithm_config, _policy_mode = get_model_info(self.model_args['algo'])
        self.model_args['policy_mode'] = _policy_mode
        if self.model_args['algo_config'] is not None:
            algorithm_config = UpdateConfig(algorithm_config, self.model_args['algo_config'], 'algo')
        algorithm_config['use_rnn'] = self.model_args['use_rnn']
        ShowConfig(algorithm_config)

        # BUFFER
        if _policy_mode == 'off-policy':
            if algorithm_config['use_rnn'] == True:
                self.buffer_args['type'] = 'EpisodeER'
                self.buffer_args['batch_size'] = algorithm_config.get('episode_batch_size', 0)
                self.buffer_args['buffer_size'] = algorithm_config.get('episode_buffer_size', 0)

                self.buffer_args['EpisodeER']['burn_in_time_step'] = algorithm_config.get('burn_in_time_step', 0)
                self.buffer_args['EpisodeER']['train_time_step'] = algorithm_config.get('train_time_step', 0)
            else:
                self.buffer_args['batch_size'] = algorithm_config.get('batch_size', 0)
                self.buffer_args['buffer_size'] = algorithm_config.get('buffer_size', 0)

                _use_priority = algorithm_config.get('use_priority', False)
                _n_step = algorithm_config.get('n_step', False)
                if _use_priority and _n_step:
                    self.buffer_args['type'] = 'NstepPER'
                    self.buffer_args['NstepPER']['max_episode'] = self.train_args['max_episode']
                    self.buffer_args['NstepPER']['gamma'] = algorithm_config['gamma']
                    algorithm_config['gamma'] = pow(algorithm_config['gamma'], self.buffer_args['NstepPER']['n'])  # update gamma for n-step training.
                elif _use_priority:
                    self.buffer_args['type'] = 'PER'
                    self.buffer_args['PER']['max_episode'] = self.train_args['max_episode']
                elif _n_step:
                    self.buffer_args['type'] = 'NstepER'
                    self.buffer_args['NstepER']['gamma'] = algorithm_config['gamma']
                    algorithm_config['gamma'] = pow(algorithm_config['gamma'], self.buffer_args['NstepER']['n'])
                else:
                    self.buffer_args['type'] = 'ER'
        else:
            self.buffer_args['type'] = 'None'
            self.train_args['pre_fill_steps'] = 0  # if on-policy, prefill experience replay is no longer needed.

        # MODEL
        base_dir = os.path.join(self.train_args['base_dir'], self.train_args['name'])  # train_args['base_dir'] DIR/ENV_NAME/ALGORITHM_NAME

        if self.env_args['type'] == 'gym':
            if self.train_args['use_wandb']:
                import wandb
                wandb_path = os.path.join(base_dir, 'wandb')
                if not os.path.exists(wandb_path):
                    os.makedirs(wandb_path)
                wandb.init(sync_tensorboard=True, name=self.train_args['name'], dir=base_dir, project=self.train_args['wandb_project'])

            # buffer ------------------------------
            if 'Nstep' in self.buffer_args['type'] or 'Episode' in self.buffer_args['type']:
                self.buffer_args[self.buffer_args['type']]['agents_num'] = self.env_args['env_num']
            self.buffer = get_buffer(self.buffer_args)
            # buffer ------------------------------

            # model -------------------------------
            model_params = {
                's_dim': self.env.s_dim,
                'visual_sources': self.env.visual_sources,
                'visual_resolution': self.env.visual_resolution,
                'a_dim': self.env.a_dim,
                'is_continuous': self.env.is_continuous,
                'max_episode': self.train_args.max_episode,
                'base_dir': base_dir,
                'logger2file': self.model_args.logger2file,
                'seed': self.model_args.seed,
                'n_agents': self.env.n
            }
            self.model = Model(**model_params, **algorithm_config)
            self.model.set_buffer(self.buffer)
            self.model.init_or_restore(self.train_args['load_model_path'])
            # model -------------------------------

            self.train_args['begin_episode'] = self.model.get_init_episode()
            if not self.train_args['inference']:
                records_dict = {
                    'env': self.env_args.to_dict,
                    'model': self.model_args.to_dict,
                    'buffer': self.buffer_args.to_dict,
                    'train': self.train_args.to_dict,
                    'algo': algorithm_config
                }
                save_config(os.path.join(base_dir, 'config'), records_dict)
                if self.train_args['use_wandb']:
                    wandb.config.update(records_dict)
        else:
            # buffer -----------------------------------
            self.buffer_args_s = []
            for i in range(self.env.brain_num):
                _bargs = deepcopy(self.buffer_args)
                if 'Nstep' in _bargs['type']or 'Episode' in _bargs['type']:
                    _bargs[_bargs['type']]['agents_num'] = self.env.brain_agents[i]
                self.buffer_args_s.append(_bargs)
            buffers = [get_buffer(self.buffer_args_s[i]) for i in range(self.env.brain_num)]
            # buffer -----------------------------------

            # model ------------------------------------
            self.model_args_s = []
            for i in range(self.env.brain_num):
                _margs = deepcopy(self.model_args)
                _margs['seed'] = self.model_args['seed'] + i * 10
                self.model_args_s.append(_margs)
            model_params = [{
                's_dim': self.env.s_dim[i],
                'a_dim': self.env.a_dim[i],
                'visual_sources': self.env.visual_sources[i],
                'visual_resolution': self.env.visual_resolutions[i],
                'is_continuous': self.env.is_continuous[i],
                'max_episode': self.train_args.max_episode,
                'base_dir': os.path.join(base_dir, b),
                'logger2file': self.model_args_s[i].logger2file,
                'seed': self.model_args_s[i].seed,    # 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
                'n_agents': self.env.brain_agents[i],
            } for i, b in enumerate(self.env.fixed_brain_names)]

            # multi agent training------------------------------------
            if self.model_args['algo'][:3] == 'ma_':
                self.ma = True
                assert self.env.brain_num > 1, 'if using ma* algorithms, number of brains must larger than 1'
                self.ma_data = ExperienceReplay(batch_size=10, capacity=1000)
                [mp.update({'n': self.env.brain_num, 'i': i}) for i, mp in enumerate(model_params)]
            else:
                self.ma = False
            # multi agent training------------------------------------

            self.models = [Model(
                **model_params[i],
                **algorithm_config
            ) for i in range(self.env.brain_num)]

            [model.set_buffer(buffer) for model, buffer in zip(self.models, buffers)]
            [self.models[i].init_or_restore(
                os.path.join(self.train_args['load_model_path'], b))
             for i, b in enumerate(self.env.fixed_brain_names)]
            # model ------------------------------------
            self.train_args['begin_episode'] = self.models[0].get_init_episode()
            if not self.train_args['inference']:
                for i, b in enumerate(self.env.fixed_brain_names):
                    records_dict = {
                        'env': self.env_args.to_dict,
                        'model': self.model_args_s[i].to_dict,
                        'buffer': self.buffer_args_s[i].to_dict,
                        'train': self.train_args.to_dict,
                        'algo': algorithm_config
                    }
                    save_config(os.path.join(base_dir, b, 'config'), records_dict)
        pass

    def pwi(self, *args, out_time=False):
        if self.all_learner_print:
            model_info = f'| Model-{self.model_index} |'
        elif int(self.model_index) == 0:
            model_info = f'|#ONLY#Model-{self.model_index} |'
        if out_time:
            model_info += f"Pass time(h:m:s) {get_time_hhmmss(self.start_time)} |"
        logger.info(
            ''.join([model_info, *args])
        )

    def __call__(self):
        self.train()

    def train(self):
        if self.env_args['type'] == 'gym':
            try:
                gym_no_op(
                    env=self.env,
                    model=self.model,
                    print_func=self.pwi,
                    pre_fill_steps=int(self.train_args['pre_fill_steps']),
                    prefill_choose=bool(self.train_args['prefill_choose'])
                )
                gym_train(
                    env=self.env,
                    model=self.model,
                    print_func=self.pwi,
                    begin_episode=int(self.train_args['begin_episode']),
                    render=bool(self.train_args['render']),
                    render_episode=int(self.train_args.get('render_episode', 50000)),
                    save_frequency=int(self.train_args['save_frequency']),
                    max_step=int(self.train_args['max_step']),
                    max_episode=int(self.train_args['max_episode']),
                    eval_while_train=bool(self.train_args['eval_while_train']),
                    max_eval_episode=int(self.train_args.get('max_eval_episode')),
                    off_policy_step_eval=bool(self.train_args['off_policy_step_eval']),
                    off_policy_step_eval_num=int(self.train_args.get('off_policy_step_eval_num')),
                    policy_mode=str(self.model_args['policy_mode']),
                    moving_average_episode=int(self.train_args['moving_average_episode']),
                    add_noise2buffer=bool(self.train_args['add_noise2buffer']),
                    add_noise2buffer_episode_interval=int(self.train_args['add_noise2buffer_episode_interval']),
                    add_noise2buffer_steps=int(self.train_args['add_noise2buffer_steps']),
                    eval_interval=int(self.train_args['eval_interval']),
                    max_learn_step=int(self.train_args['max_learn_step']),
                    max_frame_step=int(self.train_args['max_frame_step'])
                )
            finally:
                self.model.close()
                self.env.close()
        else:
            try:
                if self.ma:
                    ma_unity_no_op(
                        env=self.env,
                        models=self.models,
                        buffer=self.ma_data,
                        print_func=self.pwi,
                        pre_fill_steps=int(self.train_args['pre_fill_steps']),
                        prefill_choose=bool(self.train_args['prefill_choose'])
                    )
                    ma_unity_train(
                        env=self.env,
                        models=self.models,
                        buffer=self.ma_data,
                        print_func=self.pwi,
                        begin_episode=int(self.train_args['begin_episode']),
                        save_frequency=int(self.train_args['save_frequency']),
                        max_step=int(self.train_args['max_step']),
                        max_episode=int(self.train_args['max_episode']),
                        policy_mode=str(self.model_args['policy_mode'])
                    )
                else:
                    unity_no_op(
                        env=self.env,
                        models=self.models,
                        print_func=self.pwi,
                        pre_fill_steps=int(self.train_args['pre_fill_steps']),
                        prefill_choose=bool(self.train_args['prefill_choose']),
                        real_done=bool(self.train_args['real_done'])
                    )
                    unity_train(
                        env=self.env,
                        models=self.models,
                        print_func=self.pwi,
                        begin_episode=int(self.train_args['begin_episode']),
                        save_frequency=int(self.train_args['save_frequency']),
                        max_step=int(self.train_args['max_step']),
                        max_episode=int(self.train_args['max_episode']),
                        policy_mode=str(self.model_args['policy_mode']),
                        moving_average_episode=int(self.train_args['moving_average_episode']),
                        add_noise2buffer=bool(self.train_args['add_noise2buffer']),
                        add_noise2buffer_episode_interval=int(self.train_args['add_noise2buffer_episode_interval']),
                        add_noise2buffer_steps=int(self.train_args['add_noise2buffer_steps']),
                        total_step_control=bool(self.train_args['total_step_control']),
                        max_learn_step=int(self.train_args['max_learn_step']),
                        max_frame_step=int(self.train_args['max_frame_step']),
                        real_done=bool(self.train_args['real_done'])
                    )
            finally:
                [model.close() for model in self.models]
                self.env.close()

    def evaluate(self):
        if self.env_args['type'] == 'gym':
            gym_inference(
                env=self.env,
                model=self.model
            )
        else:
            if self.ma:
                ma_unity_inference(
                    env=self.env,
                    models=self.models
                )
            else:
                unity_inference(
                    env=self.env,
                    models=self.models
                )

    def run(self, mode='worker'):
        if mode == 'worker':
            ApexWorker(self.env, self.models)()
        elif mode == 'learner':
            ApexLearner(self.models)()
        elif mode == 'buffer':
            ApexBuffer()()
        else:
            raise Exception('unknown mode')
