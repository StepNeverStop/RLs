import gym
import time
from gym import envs
from gym.spaces import *
# __all__ = ["Space", "Box", "Discrete", "MultiDiscrete", "MultiBinary", "Tuple", "Dict", "flatdim", "flatten", "unflatten"]
env_list = envs.registry.all()
start_time = time.time()
available_file = open('available_envs.txt', 'a')
trainable_file = open('trainable_envs.txt', 'a')
untrainable_file = open('untrainable_envs.txt', 'a')
print(len(env_list))
for i, env_info in enumerate(env_list):
    try:
        if 'Defender' in env_info.id:   # Defender envs could make program no response.
            continue
        env = gym.make(env_info.id)
        info = '|' + str(env_info.id).ljust(50) + str(env.observation_space).ljust(80) + str(env.action_space).rjust(80) + '|'
        print(i, info)
        available_file.write(info + '\n')
        if type(env.observation_space) not in [Box, Discrete] or type(env.action_space) not in [Box, Discrete] or len(env.action_space.shape) == 2 or len(env.observation_space.shape) == 2:
            untrainable_file.write(info + '\n')
        else:
            trainable_file.write(info + '\n')
        env.close()
    except Exception as e:
        pass

available_file.close()
trainable_file.close()
untrainable_file.close()
end_time = time.time()
print('time_cose: ', end_time - start_time)
