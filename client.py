import rpyc
import sys
import os
import time
import socket
from rpyc import Service
import platform
import Algorithms
from mlagents.envs import UnityEnvironment
from utils.sth import sth
import pandas as pd
import zipfile
import numpy as np
from threading import Timer

_global_judge_flag = False
_global_myid = 'None'
_global_push_model = False


def create_dir(_dir):
    if os.path.isdir(_dir):
        return
    create_dir(os.path.dirname(_dir))
    os.makedirs(_dir)


def change_judge_flag():
    global _global_judge_flag
    _global_judge_flag = True


def change_push_model_flag():
    global _global_push_model
    _global_push_model = True


class ClientServer(Service):

    def exposed_set_judge_flag(self, num=None):
        if num:
            Timer(num, change_judge_flag).start()
        else:
            change_judge_flag()

    def exposed_set_push_model_flag(self):
        change_push_model_flag()

    def exposed_set_id(self, _id):
        global _global_myid
        _global_myid = _id

    def exposed_get_file_from_server(self, _root, filename, _file):
        filepath = fix_path(filename)
        root = fix_path(_root)
        if not os.path.isdir(root):
            os.makedirs(root)

        if os.path.exists(filepath):
            return
        else:
            local_file = open(filepath, 'wb')
            chunk = _file.read(1024 * 1024)
            while chunk:
                local_file.write(chunk)
                chunk = _file.read(1024 * 1024)
            local_file.close()

    def exposed_create_dir(self, _dir):
        dirpath = fix_path(_dir)
        create_dir(dirpath)

    def exposed_get_zipfile(self, filename, _file):
        filepath = fix_path(filename)
        if os.path.exists(filepath):
            return
        create_dir(os.path.dirname(filepath))
        local_file = open(filepath, 'wb')
        chunk = _file.read(1024 * 1024)
        while chunk:
            local_file.write(chunk)
            chunk = _file.read(1024 * 1024)
        local_file.close()
        f = zipfile.ZipFile(filepath, 'r')
        for _file in f.namelist():
            print(_file)
            f.extract(_file, os.path.split(filepath)[0])


def push_model(conn, model_dir):
    conn.root.create_dir(model_dir)
    for root, dirs, files in os.walk(model_dir):
        for _file in files:
            _file_path = os.path.join(root, _file)
            f_open = open(_file_path, 'rb')
            conn.root.get_file_from_client(root, _file_path, f_open)
            f_open.close()


def get_connect_option(conn):
    while True:
        print(conn.root.get_connect_info())
        item = input('plz select the number: ')
        if item == '1':
            return 'train'
        elif item == '2':
            return 'new'
        elif item == 'q':
            return 'exit'


def get_train_option(conn):
    while True:
        print(conn.root.get_training_list())
        item = input('plz select the task number that you want to trian: ')
        if item == 'q':
            return 'back'
        elif int(item) >= 0:
            return int(item)


def fix_path(filename):
    if platform.system() == "Windows":
        if ':' in filename:
            return filename.replace('\\', '/').replace(r'//', r'/')
        else:
            return 'C:' + filename
    else:
        if ':' in filename:
            return filename.replace('\\', '/').replace(r'//', r'/').split(':')[-1]
        else:
            return filename


def download_file(conn, remote_filename, local_filename):
    for i, j in zip(remote_filename, local_filename):
        block_size = 1024 * 1024
        file_size, remote_file = conn.root.open(i)
        print(str(file_size // block_size) + 'M')
        local_file = open(j.replace('test2', 'test'), 'wb')
        start = time.time()
        chunk = remote_file.read(block_size)
        acc_size = block_size
        while chunk:
            local_file.write(chunk)
            print(f'download: {acc_size/file_size:.2%}', end='\r')
            chunk = remote_file.read(block_size)
            acc_size += block_size
        print(f'cost time: {time.time()-start}')
        remote_file.close()
        local_file.close()


def initialize_env_model(filepath, algo, name, port):
    env = UnityEnvironment(
        file_name=filepath,
        base_port=port,
        no_graphics=True
    )
    if algo == 'pg':
        algorithm_config = Algorithms.pg_config
        model = Algorithms.PG
        policy_mode = 'ON'
    elif algo == 'ppo':
        algorithm_config = Algorithms.ppo_config
        model = Algorithms.PPO
        policy_mode = 'ON'
    elif algo == 'ddpg':
        algorithm_config = Algorithms.ddpg_config
        model = Algorithms.DDPG
        policy_mode = 'OFF'
    elif algo == 'td3':
        algorithm_config = Algorithms.td3_config
        model = Algorithms.TD3
        policy_mode = 'OFF'
    elif algo == 'sac':
        algorithm_config = Algorithms.sac_config
        model = Algorithms.SAC
        policy_mode = 'OFF'
    elif algo == 'sac_no_v':
        algorithm_config = Algorithms.sac_no_v_config
        model = Algorithms.SAC_NO_V
        policy_mode = 'OFF'
    else:
        raise Exception("Don't have this algorithm.")
    env_dir = os.path.split(filepath)[0]
    sys.path.append(env_dir)
    import env_config
    reset_config = env_config.reset_config
    max_step = env_config.max_step
    env_name = os.path.join(*env_dir.replace('\\', '/').replace(r'//', r'/').split('/')[-2:])
    base_dir = os.path.join(r'C:/RLData'if platform.system() == "Windows" else r'/RLData', env_name, algo, name)
    brain_names = env.external_brain_names
    brains = env.brains
    models = [model(
        s_dim=brains[i].vector_observation_space_size * brains[i].num_stacked_vector_observations,
        a_counts=brains[i].vector_action_space_size[0],
        action_type=brains[i].vector_action_space_type,
        cp_dir=os.path.join(base_dir, i, 'model'),
        log_dir=os.path.join(base_dir, i, 'log'),
        excel_dir=os.path.join(base_dir, i, 'excel'),
        logger2file=False,
        out_graph=False,
        **algorithm_config
    ) for i in brain_names]
    [sth.save_config(os.path.join(base_dir, i, 'config'), algorithm_config) for i in brain_names]

    begin_episode = models[0].get_init_step()
    max_episode = models[0].get_max_episode()
    return env, brain_names, models, policy_mode, reset_config, max_step


def run(conn):
    base_dir = r'C:/RLdata' if platform.system() == "Windows" else r'/RLData'
    while True:
        connect_option = get_connect_option(conn)
        global _global_myid
        myID = _global_myid
        if connect_option == 'exit':
            return
        if connect_option == 'train':
            train_option = get_train_option(conn)
            if train_option == 'back':
                pass
            else:
                name, _file_path, algo = conn.root.get_train_config(train_option)
                file_path = fix_path(_file_path)
                env_name = os.path.join(*os.path.split(file_path)[0].split('/')[-2:])
                model_dir = os.path.join(base_dir, env_name, algo, name)
                conn.root.get_env(myID, name)
                conn.root.get_model(myID, name, False)
                try:
                    env, brain_names, models, policy_mode, reset_config, max_step = initialize_env_model(file_path, train_config.algo, train_config.name, port=6666)
                except Exception as e:
                    print(e)
                else:
                    begin_episode = 0
                    max_episode = models[0].get_max_episode()

        elif connect_option == 'new':
            my_filepath = input('Plz input the abs path of your .exe file: ')
            zip_filepath = os.path.split(my_filepath)[0] + '.zip'
            start = time.time()
            f_open = open(zip_filepath, 'rb')
            zip_exist_flag = conn.root.push_zipfile(zip_filepath, f_open)
            f_open.close()
            # if zip_exist_flag:
            #     raise Exception('this task is already exist.')
            print(f'upload cost time: {time.time()-start}')

            algo = input('Upload success. plz input the algorithm name: ')
            port = int(input('plz input the training port: '))
            name = input('plz input the training name: ')
            save_frequency = int(input('plz input the save frequency: '))
            max_step = int(input('plz input the max_step: '))
            judge_interval = int(input('plz input the judge interval(seconds): '))
            # algo = 'sac'
            # port = 5111
            # name = 'testdis7'
            # save_frequency = 10
            # max_step = 200
            # judge_interval = 20
            try:
                env, brain_names, models, policy_mode, reset_config, max_step = initialize_env_model(my_filepath, algo, name, port)
            except Exception as e:
                print(e)
            else:
                conn.root.push_train_config(myID, name, my_filepath, algo, policy_mode, save_frequency, max_step, judge_interval)
                env_name = os.path.join(*fix_path(os.path.split(my_filepath)[0]).split('/')[-2:])
                model_dir = os.path.join(base_dir, env_name, algo, name)
                push_model(conn, model_dir)
                begin_episode = models[0].get_init_step()
                max_episode = models[0].get_max_episode()
        train(
            policy_mode=policy_mode,
            env=env,
            brain_names=brain_names,
            models=models,
            begin_episode=begin_episode,
            save_frequency=save_frequency,
            reset_config=reset_config,
            max_step=max_step,
            max_episode=max_episode,
            conn=conn,
            myID=myID,
            name=name,
            model_dir=model_dir,
        )


def train(
        policy_mode,
        env,
        brain_names,
        models,
        begin_episode,
        save_frequency,
        reset_config,
        max_step,
        max_episode,
        conn,
        myID,
        name,
        model_dir
):

    global _global_push_model
    train_func = on_train if policy_mode == 'ON' else off_train
    while True:
        begin_episode, models_global_step, ave_reward = train_func(
            env=env,
            brain_names=brain_names,
            models=models,
            begin_episode=begin_episode,
            save_frequency=save_frequency,
            reset_config=reset_config,
            max_step=max_step,
            max_episode=max_episode
        )
        start = time.time()
        conn.root.push_reward(myID, ave_reward)
        while True:
            if _global_push_model:
                _global_push_model = False
                print(True)
                push_model(conn, model_dir)
                break
        conn.root.get_model(myID, name)
        print(f'cost time: {time.time()-start}')
        for i, brain_name in enumerate(brain_names):
            models[i].init_or_restore(os.path.join(model_dir, brain_name, 'model'))
            models[i].set_global_step(models_global_step[i])


def on_train(
    env,
    brain_names,
    models,
    begin_episode,
    save_frequency,
    reset_config,
    max_step,
    max_episode
):
    ave_reward_list = []
    brains_num = len(brain_names)
    state = [0] * brains_num
    action = [0] * brains_num
    dones_flag = [0] * brains_num
    agents_num = [0] * brains_num
    total_reward = [0] * brains_num

    for episode in range(begin_episode, max_episode):
        global _global_judge_flag
        if _global_judge_flag:
            _global_judge_flag = False
            models_global_step = [models[i].get_global_step() for i in range(brains_num)]
            ave_reward = np.array(ave_reward_list[-(len(ave_reward_list) // 4):]).mean()
            return episode, models_global_step, ave_reward

        obs = env.reset(config=reset_config, train_mode=True)
        for i, brain_name in enumerate(brain_names):
            agents_num[i] = len(obs[brain_name].agents)
            dones_flag[i] = np.zeros(agents_num[i])
            total_reward[i] = np.zeros(agents_num[i])

        step = 0

        while True:
            step += 1

            for i, brain_name in enumerate(brain_names):
                state[i] = obs[brain_name].vector_observations
                action[i] = models[i].choose_action(s=state[i])

            actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(brain_names)}
            obs = env.step(vector_action=actions)

            for i, brain_name in enumerate(brain_names):
                dones_flag[i] += obs[brain_name].local_done
                models[i].store_data(
                    s=state[i],
                    a=action[i],
                    r=np.array(obs[brain_name].rewards),
                    s_=obs[brain_name].vector_observations,
                    done=np.array(obs[brain_name].local_done)
                )
                total_reward[i] += np.array(obs[brain_name].rewards)

            if all([all(dones_flag[i]) for i in range(brains_num)]) or step > max_step:
                for i in range(brains_num):
                    models[i].learn(episode)
                    models[i].writer_summary(episode)
                break
        ave_reward_list.append(np.array([total_reward[i].mean() for i in range(brains_num)]).mean())
        print(f'episode {episode} step {step}')
        if episode % save_frequency == 0:
            for i in range(brains_num):
                models[i].save_checkpoint(episode)


def off_train(
    env,
    brain_names,
    models,
    begin_episode,
    save_frequency,
    reset_config,
    max_step,
    max_episode
):
    global _global_judge_flag
    ave_reward_list = []
    brains_num = len(brain_names)
    state = [0] * brains_num
    action = [0] * brains_num
    dones_flag = [0] * brains_num
    agents_num = [0] * brains_num
    total_reward = [0] * brains_num

    for episode in range(begin_episode, max_episode):

        obs = env.reset(config=reset_config, train_mode=True)
        for i, brain_name in enumerate(brain_names):
            agents_num[i] = len(obs[brain_name].agents)
            dones_flag[i] = np.zeros(agents_num[i])
            total_reward[i] = np.zeros(agents_num[i])

        step = 0

        while True:
            step += 1

            if _global_judge_flag:
                _global_judge_flag = False
                models_global_step = [models[i].get_global_step() for i in range(brains_num)]
                ave_reward = np.array(ave_reward_list[-(len(ave_reward_list) // 4):]).mean()
                return episode, models_global_step, float(ave_reward)

            for i, brain_name in enumerate(brain_names):
                state[i] = obs[brain_name].vector_observations
                action[i] = models[i].choose_action(s=state[i])

            actions = {f'{brain_name}': action[i] for i, brain_name in enumerate(brain_names)}
            obs = env.step(vector_action=actions)

            for i, brain_name in enumerate(brain_names):
                dones_flag[i] += obs[brain_name].local_done
                models[i].store_data(
                    s=state[i],
                    a=action[i],
                    r=np.array(obs[brain_name].rewards)[:, np.newaxis],
                    s_=obs[brain_name].vector_observations,
                    done=np.array(obs[brain_name].local_done)[:, np.newaxis]
                )
                models[i].learn(episode)
            if all([all(dones_flag[i]) for i in range(brains_num)]) or step > max_step:
                break
        ave_reward_list.append(np.array([total_reward[i].mean() for i in range(brains_num)]).mean())
        print(f'episode {episode} step {step}')
        for i in range(brains_num):
            models[i].writer_summary(episode)
        if episode % save_frequency == 0:
            for i in range(brains_num):
                models[i].save_checkpoint(episode)


if __name__ == "__main__":
    conn = rpyc.connect(
        host='111.186.116.71',
        port=12345,
        keepalive=True,
        service=ClientServer,
        config={
            'allow_public_attrs': True,
            'sync_request_timeout': 120
        }
    )
    try:
        run(conn)
    except Exception as e:
        print(e)
    finally:
        conn.close()
        sys.exit()
