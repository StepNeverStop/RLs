# coding: utf-8
"""
Usage:
    python [options]

Options:
    -h,--help                   显示帮助
    -i,--inference              推断 [default: False]
    -a,--algorithm=<name>       算法 [default: ppo]
    -c,--config-file=<file>     指定模型的超参数config文件 [default: None]
    -e,--env=<file>             指定环境名称 [default: None]
    -p,--port=<n>               端口 [default: 5005]
    -u,--unity                  是否使用unity客户端 [default: False]
    -g,--graphic                是否显示图形界面 [default: False]
    -m,--model-string=<name>    训练的名字/载入模型的路径 [default: None]
    -s,--save-frequency=<n>     保存频率 [default: None]
Example:
    python run.py -a sac -g -e C:/test.exe -p 6666 -s 10 -m test
"""
import os
import sys
import Algorithms
from docopt import docopt
from config import train_config
from mlagents.envs import UnityEnvironment


def run():
    options = docopt(__doc__)
    print(options)
    reset_config=train_config['reset_config']
    save_frequency=train_config['save_frequency'] if options['--save-frequency'] =='None' else int(options['--save-frequency']) 
    model_string = train_config['model_string'] if options['--model-string']=='None' else options['--model-string']
    if options['--env'] != 'None':
        file_name = options['--env']
    else:
        file_name=train_config['exe_file']

    if options['--unity']:
        file_name=None
    
    if file_name!=None:
        if os.path.exists(file_name):
            env=UnityEnvironment(
                file_name=file_name,
                base_port=int(options['--port']),
                no_graphics= False if options['--inference'] else not options['--graphic']
            )
            env_dir = os.path.split(
                options['--env'])[0]
            env_name = env_dir.split('UnityBuild')[-1]
            sys.path.append(env_dir)
            if os.path.exists(env_dir + '/env_config.py'):
                import env_config
                reset_config = env_config.reset_config
            if os.path.exists(env_dir + '/env_loop.py'):
                from env_loop import Loop
        else:
            raise Exception('can not find this file.')
    else:
        env = UnityEnvironment()
        env_name='/unity'

    if options['--algorithm'] == 'ppo':
        algorithm_config = Algorithms.ppo_config
        model = Algorithms.PPO
        policy_mode = 'ON'
    elif options['--algorithm'] == 'ddpg':
        algorithm_config = Algorithms.ddpg_config
        model = Algorithms.DDPG
        policy_mode = 'OFF'
    elif options['--algorithm'] == 'td3':
        algorithm_config = Algorithms.td3_config
        model = Algorithms.TD3
        policy_mode = 'OFF'
    elif options['--algorithm'] == 'sac':
        algorithm_config = Algorithms.sac_config
        model = Algorithms.SAC
        policy_mode = 'OFF'
    elif options['--algorithm'] == 'sac_no_v':
        algorithm_config = Algorithms.sac_no_v_config
        model = Algorithms.SAC_NO_V
        policy_mode = 'OFF'
    else:
        raise Exception("Don't have this algorithm.")

    # if option['--config-file'] != 'None' and os.path.exists(option['--config-file']):

        
    if 'Loop' not in locals().keys():
        from loop import Loop
    base_dir = train_config['base_dir']+env_name+'/'+options['--algorithm']+'/'+model_string+'/'

    brain_names = env.external_brain_names
    brains = env.brains
    models = [model(
        s_dim=brains[i].vector_observation_space_size,
        a_counts=brains[i].vector_action_space_size[0],
        cp_dir=base_dir + f'{i}'+'/model/',
        log_dir=base_dir + f'{i}'+'/log/',
        excel_dir=base_dir + f'{i}'+'/excel/',
        logger2file=False,
        out_graph=False,
        **algorithm_config
    ) for i in brain_names]

    begin_episode = models[0].get_init_step(
        cp_dir=base_dir + brain_names[0]+'/model/')
    if options['--inference']:
        Loop.inference(env, brain_names, models, reset_config=reset_config)
    else:
        if policy_mode == 'ON':
            Loop.train_OnPolicy(env, brain_names, models,
                                begin_episode, save_frequency=save_frequency, reset_config=reset_config)
        else:
            Loop.train_OffPolicy(env, brain_names, models,
                                 begin_episode, save_frequency=save_frequency, reset_config=reset_config)


if __name__ == "__main__":
    run()
