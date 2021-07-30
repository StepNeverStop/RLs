import importlib
import pettingzoo
import numpy as np

from copy import deepcopy
from typing import (List,
                    NoReturn)
from collections import defaultdict
from gym.spaces import (Box,
                        Discrete,
                        Tuple)

from rls.envs.env_base import EnvBase
from rls.common.specs import (ObsSpec,
                              EnvGroupArgs,
                              ModelObservations,
                              SingleModelInformation,
                              generate_obs_dataformat)
from rls.envs.pettingzoo.wrappers import BasicWrapper


def get_vectorized_env_class(vector_env_type):
    '''Import gym env vectorize wrapper class'''
    if vector_env_type == 'multiprocessing':
        from rls.envs.gym.wrappers.multiprocessing_wrapper import MultiProcessingEnv as AsynVectorEnvClass
    elif vector_env_type == 'multithreading':
        from rls.envs.gym.wrappers.threading_wrapper import MultiThreadEnv as AsynVectorEnvClass
    elif vector_env_type == 'ray':
        from rls.envs.gym.wrappers.ray_wrapper import RayEnv as AsynVectorEnvClass
    elif vector_env_type == 'vector':
        from rls.envs.gym.wrappers.vector_wrapper import VectorEnv as AsynVectorEnvClass
    else:
        raise Exception('The vector_env_type doesn\'in the list of [multiprocessing, multithreading, ray, vector]. Please check your configurations.')

    return AsynVectorEnvClass


class PettingZooEnv(EnvBase):

    def __init__(self,
                 env_copys=1,
                 seed=42,
                 vector_env_type='vector',
                 env_name='mpe.simple_v2',
                 env_config={},
                 **kwargs):

        self._n_copys = env_copys   # environments number

        # NOTE: env_name should be formatted like `mpe.simple_v2`
        env_module = importlib.import_module(f"pettingzoo.{env_name}")
        env_module = env_module.parallel_env
        def make_env(idx, **config): return BasicWrapper(env_module(**config))  # TODO:

        self._initialize(env=make_env(0, **env_config))
        self._envs = get_vectorized_env_class(vector_env_type)(make_env, env_config, self._n_copys, seed)

    def reset(self, **kwargs) -> List[ModelObservations]:
        ret = self._envs.reset()
        obs = defaultdict(list)
        for k in self._agents:
            for _ret in ret:
                obs[k].append(_ret[k])
            obs[k] = np.asarray(obs[k])

        return [ModelObservations(vector=self.vector_info_type[k](*(obs[k],)),
                                  visual=self.visual_info_type[k](*(obs[k],)))
                for k in self._agents]

    def step(self, actions: List[np.ndarray], **kwargs) -> List[SingleModelInformation]:
        actions = np.asarray(actions)   # [N, B, *]
        actions = actions.swapaxes(0, 1)    # [B, N, *]
        new_actions = []
        for action in actions:
            new_actions.append(dict(list(zip(self._agents, action))))

        results = self._envs.step(new_actions)  # [B, *]

        obs = defaultdict(list)  # [N, B, *]
        reward = defaultdict(list)
        done = defaultdict(list)
        info = defaultdict(list)
        for k in self._agents:
            for _obs, _reward, _done, _info in results:
                obs[k].append(_obs[k])
                reward[k].append(_reward[k])
                done[k].append(_done[k])
                info[k].append(_info[k])   # TODO
            obs[k] = np.asarray(obs[k])  # [B, *]
            reward[k] = np.asarray(reward[k]).astype('float32')
            done[k] = np.asarray(done[k])
            info[k] = np.asarray(info[k])   # TODO

        correct_new_obs = deepcopy(obs)

        dones_flag = np.asarray(list(done.values())).any(0)  # [N, B] => [B, ]
        if dones_flag.any():
            dones_index = np.where(dones_flag)[0]
            ret = self._envs.reset(dones_index.tolist())
            reset_obs = defaultdict(list)   # [N, B, *]
            for k in self._agents:
                for _ret in ret:
                    reset_obs[k].append(_ret[k])
                correct_new_obs[k][dones_index] = np.asarray(reset_obs[k])

        return [SingleModelInformation(corrected_obs=ModelObservations(vector=self.vector_info_type[k](*(correct_new_obs[k],)),
                                                                       visual=self.visual_info_type[k](*(correct_new_obs[k],))),
                                       obs=ModelObservations(vector=self.vector_info_type[k](*(obs[k],)),
                                                             visual=self.visual_info_type[k](*(obs[k],))),
                                       reward=reward[k],
                                       done=done[k],
                                       info=info[k])
                for k in self._agents]

    def close(self, **kwargs) -> NoReturn:
        self._envs.close()

    def random_action(self, **kwargs) -> List[np.ndarray]:
        ret = self._envs.action_sample()
        actions = defaultdict(list)
        for k in self._agents:
            for _ret in ret:
                actions[k].append(_ret[k])
            actions[k] = np.asarray(actions[k])
        return list(actions.values())

    def render(self, **kwargs) -> NoReturn:
        record = kwargs.get('record', False)
        self._envs.render(record, [0])

    @property
    def n_agents(self) -> int:
        return len(self._agents)

    @property
    def n_copys(self) -> int:
        return self._n_copys

    @property
    def GroupsSpec(self) -> List[EnvGroupArgs]:
        return [EnvGroupArgs(
            obs_spec=ObsSpec(vector_dims=self.vector_dims[k],
                             visual_dims=self.visual_dims[k]),
            a_dim=self.a_dim[k],
            is_continuous=self._is_continuous[k],
            n_copys=self._n_copys
        ) for k in self._agents]

    @property
    def is_multi(self) -> bool:
        return self.n_agents > 1

    # --- custome

    def _initialize(self, env):
        self.vector_dims = defaultdict(list)
        self.visual_dims = defaultdict(list)
        self.vector_info_type = {}
        self.visual_info_type = {}

        self.a_dim = defaultdict(int)
        self._is_continuous = {}

        self._agents = env.possible_agents
        for k in env.possible_agents:
            ObsSpace = env.observation_spaces[k]
            if isinstance(ObsSpace, Box):
                if len(ObsSpace.shape) == 1:
                    self.vector_dims[k] = [ObsSpace.shape[0]]
                else:
                    self.vector_dims[k] = []
            else:
                self.vector_dims[k] = [int(ObsSpace.n)]
            if len(ObsSpace.shape) == 3:
                self.visual_dims[k] = [list(ObsSpace.shape)]
            else:
                self.visual_dims[k] = []

            self.vector_info_type[k] = generate_obs_dataformat(n_copys=self._n_copys,
                                                               item_nums=len(self.vector_dims[k]),
                                                               name='vector')
            self.visual_info_type[k] = generate_obs_dataformat(n_copys=self._n_copys,
                                                               item_nums=len(self.visual_dims[k]),
                                                               name='visual')
            # process action
            ActSpace = env.action_spaces[k]
            if isinstance(ActSpace, Box):
                assert len(ActSpace.shape) == 1, 'if action space is continuous, the shape length of action must equal to 1'
                self._is_continuous[k] = True
                self.a_dim[k] = ActSpace.shape[0]
            elif isinstance(ActSpace, Tuple):
                assert all([isinstance(i, Discrete) for i in ActSpace]) == True, 'if action space is Tuple, each item in it must have type Discrete'
                self._is_continuous[k] = False
                self.a_dim[k] = int(np.asarray([i.n for i in ActSpace]).prod())
            else:
                self._is_continuous[k] = False
                self.a_dim[k] = ActSpace.n
        env.close()


if __name__ == '__main__':

    env = PettingZooEnv(env_copys=10,
                        seed=42,
                        vector_env_type='vector',
                        env_name='mpe.simple_adversary_v2',
                        env_config={
                            'continuous_actions': True
                        })
    print(env.GroupsSpec)
    input()
    while True:
        obs = env.reset()
        for i in range(100):
            actions = env.random_action()
            ret = env.step(actions)
            print(i, [_ret.done for _ret in ret])
