import rpyc
import sys
import os
import time
from rpyc import Service
import platform
import Algorithms
from mlagents.envs import UnityEnvironment
from common.yaml_ops import save_config
import pandas as pd
import zipfile
import numpy as np
from threading import Timer
import threading
import shutil

_global_judge_flag = False
_global_myid = 'None'


def fix_path(filename):
    if platform.system() == "Windows":
        if ':' in filename:
            return filename.replace('\\', '/').replace(r'//', r'/').replace('C:/server', 'C:')
        else:
            return 'C:' + filename.replace('/server', '')
    else:
        if ':' in filename:
            return filename.replace('\\', '/').replace(r'//', r'/').split(':/server')[-1]
        else:
            return filename.replace('/server', '')


def clear_model(model_dir):
    for i in os.listdir(model_dir):
        path_file = os.path.join(model_dir, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            clear_model(path_file)


def change_judge_flag():
    global _global_judge_flag
    _global_judge_flag = True


class ClientServer(Service):

    def exposed_set_judge_flag(self, num):
        Timer(num, change_judge_flag).start()

    def exposed_set_id(self, _id):
        global _global_myid
        _global_myid = _id

    def exposed_get_file_from_server(self, _root, filename, _file):
        filepath = fix_path(filename)
        root = fix_path(_root)
        if not os.path.isdir(root):
            os.makedirs(root)
        local_file = open(filepath, 'wb')
        chunk = _file.read(1024 * 1024)
        while chunk:
            local_file.write(chunk)
            chunk = _file.read(1024 * 1024)
        local_file.close()

    def exposed_get_zipfile(self, filename, _file):
        filepath = fix_path(filename)
        if os.path.exists(filepath):
            return
        os.makedirs(os.path.dirname(filepath))
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


def push_model(conn, job_name, model_dir):
    conn.root.clear_model(model_dir)
    for root, dirs, files in os.walk(model_dir):
        for _file in files:
            _file_path = os.path.join(root, _file)
            f_open = open(_file_path, 'rb')
            conn.root.get_file_from_client(root, _file_path, f_open)
            f_open.close()
    conn.root.set_push_done_flag(job_name)


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
    env_name = os.path.join(*fix_path(env_dir).split('/')[-2:])
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
    [save_config(os.path.join(base_dir, i, 'config'), algorithm_config) for i in brain_names]

    begin_episode = models[0].get_init_step()
    max_episode = models[0].get_max_episode()
    return env, brain_names, models, policy_mode, reset_config, max_step


def run(conn):
    base_dir = r'C:/RLData' if platform.system() == "Windows" else r'/RLData'
    while True:
        connect_option = get_connect_option(conn)
        global _global_myid
        myID = _global_myid
        if connect_option == 'exit':
            return
        if connect_option == 'train':
            train_option = get_train_option(conn)
            if train_option == 'back':
                continue
            else:
                name, _file_path, algo, save_frequency, max_step = conn.root.get_train_config(train_option)
                file_path = fix_path(_file_path)
                env_name = os.path.join(*os.path.dirname(file_path).split('/')[-2:])
                model_dir = os.path.join(base_dir, env_name, algo, name)
                conn.root.get_env(myID, name)
                conn.root.get_model(myID, name)
                try:
                    env, brain_names, models, policy_mode, reset_config, max_step = initialize_env_model(file_path, algo, name, port=6666)
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
            judge_interval = int(input('plz input the judge interval(seconds): '))
            # algo = 'sac'
            # port = 5111
            # name = 'testdis'
            # judge_interval = 60
            save_frequency = 0
            try:
                env, brain_names, models, policy_mode, reset_config, max_step = initialize_env_model(my_filepath, algo, name, port)
            except Exception as e:
                print(e)
            else:
                conn.root.push_train_config(myID, name, my_filepath, algo, policy_mode, save_frequency, max_step, judge_interval)
                env_name = os.path.join(*fix_path(os.path.split(my_filepath)[0]).split('/')[-2:])
                model_dir = os.path.join(base_dir, env_name, algo, name)
                push_model(conn, name, os.path.join(model_dir, brain_names[0], 'model'))
                begin_episode = models[0].get_init_step()
                max_episode = models[0].get_max_episode()
        threading.Thread(target=train, args=(
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
            model_dir,
            connect_option
        )).start()


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
        model_dir,
        connect_option
):
    brains_num = len(brain_names)
    conn.root.register_train_task(myID, name)
    train_func = on_train if policy_mode == 'ON' else off_train
    model_dirs = [os.path.join(model_dir, brain_name, 'model') for brain_name in brain_names]
    while True:
        conn.root.set_timer(myID, name)
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
        for i in range(brains_num):
            clear_model(model_dirs[i])
            models[i].save_checkpoint(begin_episode)
        start = time.time()
        conn.root.push_reward(myID, ave_reward)
        print('Push Reward Success.')
        while True:
            need_push_id = int(conn.root.get_need_push_id(name))
            print(need_push_id)
            if need_push_id == myID:
                for model_dir in model_dirs:
                    push_model(conn, name, model_dir)
                print('Push Model Success.')
                break
            elif need_push_id != 0:
                break
        while True:
            if conn.root.get_model_flag(name):
                conn.root.get_model(myID, name)
                break
        print(f'cost time: {time.time()-start}')
        for i in range(brains_num):
            models[i].init_or_restore(model_dirs[i])
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
    global _global_judge_flag
    ave_reward_list = [0]
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

            if _global_judge_flag:
                _global_judge_flag = False
                models_global_step = [models[i].get_global_step() for i in range(brains_num)]
                ave_reward = np.array(ave_reward_list[-(len(ave_reward_list) // 4):]).mean()
                return episode, models_global_step, ave_reward
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
        # if episode % save_frequency == 0:
        #     for i in range(brains_num):
        #         models[i].save_checkpoint(episode)


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
    ave_reward_list = [0]
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
                total_reward[i] += np.array(obs[brain_name].rewards)
                models[i].learn(episode)
            if all([all(dones_flag[i]) for i in range(brains_num)]) or step > max_step:
                break
        ave_reward_list.append(np.array([total_reward[i].mean() for i in range(brains_num)]).mean())
        print(f'episode {episode} step {step}')
        for i in range(brains_num):
            models[i].writer_summary(episode, reward=total_reward[i].mean())
        # if episode % save_frequency == 0:
        #     for i in range(brains_num):
        #         models[i].save_checkpoint(episode)


if __name__ == "__main__":
    conn = rpyc.connect(
        host='10.0.4.227',
        port=32643,
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
