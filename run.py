
#!/usr/bin/env python3
# encoding: utf-8
"""
Usage:
    python [options]

Options:
    -h,--help                   显示帮助
    -a,--algorithm=<name>       算法
                                specify the training algorithm [default: ppo]
    -c,--copys=<n>              指定并行训练的数量
                                nums of environment copys that collect data in parallel [default: 1]
    -e,--env=<file>             指定Unity环境路径
                                specify the path of builded training environment of UNITY3D [default: None]
    -g,--graphic                是否显示图形界面
                                whether show graphic interface when using UNITY3D [default: False]
    -i,--inference              推断
                                inference the trained model, not train policies [default: False]
    -m,--models=<n>             同时训练多少个模型
                                specify the number of trails that using different random seeds [default: 1]
    -n,--name=<name>            训练的名字
                                specify the name of this training task [default: None]
    -p,--port=<n>               端口
                                specify the port that communicate with training environment of UNITY3D [default: 5005]
    -r,--rnn                    是否使用RNN模型
                                whether use rnn[GRU, LSTM, ...] or not [default: False]
    -s,--save-frequency=<n>     保存频率
                                specify the interval that saving model checkpoint [default: None]
    -t,--train-step=<n>         总的训练次数
                                specify the training step that optimize the policy model [default: None]
    -u,--unity                  是否使用unity客户端
                                whether training with UNITY3D editor [default: False]
    
    --apex=<str>                i.e. "learner"/"worker"/"buffer"/"evaluator" [default: None]
    --unity-env=<name>          指定unity环境的名字
                                specify the name of training environment of UNITY3D [default: None]
    --config-file=<file>        指定模型的超参数config文件
                                specify the path of training configuration file [default: None]
    --store-dir=<file>          指定要保存模型、日志、数据的文件夹路径
                                specify the directory that store model, log and others [default: None]
    --seed=<n>                  指定训练器全局随机种子
                                specify the random seed of module random, numpy and tensorflow [default: 42]
    --unity-env-seed=<n>        指定unity环境的随机种子
                                specify the environment random seed of UNITY3D [default: 42]
    --max-step=<n>              每回合最大步长
                                specify the maximum step per episode [default: None]
    --train-episode=<n>         总的训练回合数
                                specify the training maximum episode [default: None]
    --train-frame=<n>           总的训练采样次数
                                specify the training maximum steps interacting with environment [default: None]
    --load=<name>               指定载入model的训练名称
                                specify the name of pre-trained model that need to load [default: None]
    --prefill-steps=<n>         指定预填充的经验数量
                                specify the number of experiences that should be collected before start training, use for off-policy algorithms [default: None]
    --prefill-choose            指定no_op操作时随机选择动作，或者置0
                                whether choose action using model or choose randomly [default: False]
    --gym                       是否使用gym训练环境
                                whether training with gym [default: False]
    --gym-env=<name>            指定gym环境的名字
                                specify the environment name of gym [default: CartPole-v0]
    --gym-env-seed=<n>          指定gym环境的随机种子
                                specify the environment random seed of gym [default: 42]
    --render-episode=<n>        指定gym环境从何时开始渲染
                                specify when to render the graphic interface of gym environment [default: None]
    --info=<str>                抒写该训练的描述，用双引号包裹
                                write another information that describe this training task [default: None]
    --use-wandb                 是否上传数据到W&B
                                whether upload training log to WandB [default: False]
    --hostname                  是否在训练名称后附加上主机名称
                                whether concatenate hostname with the training name [default: False]
    --no-save                 指定是否在训练中保存模型、日志及训练数据
                                specify whether save models/logs/summaries while training or not [default: False]
Example:
    gym:
        python run.py --gym -a dqn --gym-env CartPole-v0 -c 12 -n dqn_cartpole --no-save
    unity:
        python run.py -u -a ppo -n run_with_unity
        python run.py -e /root/env/3dball.app -a sac -n run_with_execution_file
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import sys

if sys.platform.startswith('win'):
    import win32api
    import win32con
    import _thread

    def _win_handler(event, hook_sigint=_thread.interrupt_main):
        '''
        handle the event of 'Ctrl+c' in windows operating system.
        '''
        if event == 0:
            hook_sigint()
            return 1
        return 0
    # Add the _win_handler function to the windows console's handler function list
    win32api.SetConsoleCtrlHandler(_win_handler, 1)

import time
import logging

from typing import Dict
from copy import deepcopy
from docopt import docopt
from multiprocessing import Process

from rls.common.trainer import Trainer
from rls.common.config import Config
from rls.common.yaml_ops import load_yaml
from rls.parse.parse_op import parse_options
from rls.utils.display import show_dict
from rls.utils.logging_utils import set_log_level
set_log_level(logging.INFO)


def get_options(options: Dict) -> Config:
    '''
    Resolves command-line arguments
    params:
        options: dictionary of command-line arguments
    return:
        op: an instance of Config class that contains the parameters
    '''
    def f(k, t): return None if options[k] == 'None' else t(options[k])
    op = Config()
    op.add_dict(dict([
        ['inference', bool(options['--inference'])],
        ['algo', str(options['--algorithm'])],
        ['use_rnn', bool(options['--rnn'])],
        ['algo_config', f('--config-file', str)],
        ['env', f('--env', str)],
        ['port', int(options['--port'])],
        ['unity', bool(options['--unity'])],
        ['graphic', bool(options['--graphic'])],
        ['name', f('--name', str)],
        ['save_frequency', f('--save-frequency', int)],
        ['models', int(options['--models'])],
        ['store_dir', f('--store-dir', str)],
        ['seed', int(options['--seed'])],
        ['unity_env_seed', int(options['--unity-env-seed'])],
        ['max_step_per_episode', f('--max-step', int)],
        ['max_train_step', f('--train-step', int)],
        ['max_train_frame', f('--train-frame', int)],
        ['max_train_episode', f('--train-episode', int)],
        ['load', f('--load', str)],
        ['prefill_steps', f('--prefill-steps', int)],
        ['prefill_choose', bool(options['--prefill-choose'])],
        ['gym', bool(options['--gym'])],
        ['n_copys', int(options['--copys'])],
        ['gym_env', str(options['--gym-env'])],
        ['gym_env_seed', int(options['--gym-env-seed'])],
        ['render_episode', f('--render-episode', int)],
        ['info', f('--info', str)],
        ['use_wandb', bool(options['--use-wandb'])],
        ['unity_env', f('--unity-env', str)],
        ['apex', f('--apex', str)],
        ['hostname', bool(options['--hostname'])],
        ['no_save', bool(options['--no-save'])]
    ]))
    return op


def agent_run(*args):
    '''
    Start a training task
    '''
    Trainer(*args)()


def main():
    options = docopt(__doc__)
    options = get_options(dict(options))
    show_dict(options.to_dict)

    trails = options.models
    assert trails > 0, '--models must greater than 0.'

    env_args, buffer_args, train_args = parse_options(options, default_config=load_yaml(f'./config.yaml'))

    if options.inference:
        Trainer(env_args, buffer_args, train_args).evaluate()
        return

    if options.apex is not None:
        train_args.update(load_yaml(f'./rls/distribute/apex/config.yaml'))
        Trainer(env_args, buffer_args, train_args).apex()
    else:
        if trails == 1:
            agent_run(env_args, buffer_args, train_args)
        elif trails > 1:
            processes = []
            for i in range(trails):
                _env_args, _buffer_args, _train_args = map(deepcopy, [env_args, buffer_args, train_args])
                _train_args.seed += i * 10
                _train_args.name += f'/{i}'
                _train_args.allow_print = True  # NOTE: set this could block other processes' print function
                if _env_args.type == 'unity':
                    _env_args.port = env_args.port + i
                p = Process(target=agent_run, args=(_env_args, _buffer_args, _train_args))
                p.start()
                time.sleep(10)
                processes.append(p)
            [p.join() for p in processes]


if __name__ == "__main__":
    try:
        import colored_traceback
        colored_traceback.add_hook()
    except ImportError:
        pass
    try:
        main()
    except Exception as e:
        print(e)
        sys.exit()
