
import numpy as np

from typing import (List,
                    NoReturn)
from gym.spaces import (Box,
                        Discrete,
                        Tuple)
from copy import deepcopy

from rls.envs.env_base import EnvBase
from rls.common.specs import (ObsSpec,
                              EnvGroupArgs,
                              ModelObservations,
                              SingleModelInformation,
                              generate_obs_dataformat)
from rls.envs.gym.make_env import make_env
from rls.utils.display import colorize
from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)

try:
    import gym_minigrid
except ImportError:
    logger.warning(colorize("import gym_minigrid failed, using 'pip3 install gym-minigrid' install it.", color='yellow'))
    pass

try:
    # if wanna render, added 'renders=True' or(depends on env) 'render=True' in gym.make() function manually.
    import pybullet_envs
except ImportError:
    logger.warning(colorize("import pybullet_envs failed, using 'pip3 install PyBullet' install it.", color='yellow'))
    pass

try:
    import gym_donkeycar
except ImportError:
    logger.warning(colorize("import gym_minigrid failed, using 'pip install gym_donkeycar' install it.", color='yellow'))
    pass

try:
    import highway_env
except ImportError:
    logger.warning(colorize("import highway_env failed, using 'pip install --user git+https://github.com/eleurent/highway-env' install it.", color='yellow'))
    pass


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


class GymEnv(EnvBase):

    def __init__(self,
                 env_copys=1,
                 render_all=False,
                 vector_env_type='vector',
                 seed=42,
                 **kwargs):
        '''
        Input:
            env_copys: environment number
        '''
        self._n_copys = env_copys   # environments number
        self.render_index = self._get_render_index(render_all)
        self._initialize(env=make_env(**kwargs))
        self._envs = get_vectorized_env_class(vector_env_type)(make_env, kwargs, self._n_copys, seed)

    def reset(self, **kwargs) -> List[ModelObservations]:
        obs = np.asarray(self._envs.reset())
        return [ModelObservations(vector=self.vector_info_type(*(obs,)),
                                  visual=self.visual_info_type(*(obs,)))]

    def step(self, actions: List[np.ndarray], **kwargs) -> List[SingleModelInformation]:
        actions = np.array(actions[0])  # choose the first agents' actions

        results = self._envs.step(actions)

        obs, reward, done, info = [np.asarray(e) for e in zip(*results)]
        reward = reward.astype('float32')
        correct_new_obs = self._partial_reset(obs, done)

        corrected_obs = ModelObservations(vector=self.vector_info_type(*(correct_new_obs,)),
                                          visual=self.visual_info_type(*(correct_new_obs,)))
        obs = ModelObservations(vector=self.vector_info_type(*(obs,)),
                                visual=self.visual_info_type(*(obs,)))

        return [SingleModelInformation(corrected_obs=corrected_obs,
                                       obs=obs,
                                       reward=reward,
                                       done=done,
                                       info=info)]

    def close(self, **kwargs) -> NoReturn:
        '''
        close all environments.
        '''
        self._envs.close()

    def random_action(self, **kwargs) -> List[np.ndarray]:
        '''
        generate random actions for all training environment.
        '''
        return [np.asarray(self._envs.action_sample())]

    def render(self, **kwargs) -> NoReturn:
        '''
        render game windows.
        '''
        record = kwargs.get('record', False)
        self._envs.render(record, self.render_index)

    @property
    def n_agents(self) -> int:
        return 1

    @property
    def n_copys(self) -> int:
        return int(self._n_copys)

    @property
    def GroupsSpec(self) -> List[EnvGroupArgs]:
        return [EnvGroupArgs(
            obs_spec=ObsSpec(vector_dims=self.vector_dims,
                             visual_dims=self.visual_dims),
            a_dim=self.a_dim,
            is_continuous=self._is_continuous,
            n_copys=self._n_copys
        )]

    @property
    def is_multi(self) -> bool:
        return self.n_agents > 1

    # --- custom

    def _get_render_index(self, render_all):
        '''
        get render windows list, i.e. [0, 1] when there are 4 training enviornment.
        '''
        assert isinstance(render_all, bool), 'assert isinstance(render_all, bool)'
        if render_all:
            render_index = [i for i in range(self._n_copys)]
        else:
            import random
            render_index = random.sample([i for i in range(self._n_copys)], 1)
        return render_index

    def _initialize(self, env):
        assert isinstance(env.observation_space, (Box, Discrete)) and isinstance(env.action_space, (Box, Discrete)), 'action_space and observation_space must be one of available_type'
        # process observation
        ObsSpace = env.observation_space
        if isinstance(ObsSpace, Box):
            if len(ObsSpace.shape) == 1:
                self.vector_dims = [ObsSpace.shape[0]]
            else:
                self.vector_dims = []
        else:
            self.vector_dims = [int(ObsSpace.n)]
        if len(ObsSpace.shape) == 3:
            self.visual_dims = [list(ObsSpace.shape)]
        else:
            self.visual_dims = []

        self.vector_info_type = generate_obs_dataformat(n_copys=self._n_copys,
                                                        item_nums=len(self.vector_dims),
                                                        name='vector')
        self.visual_info_type = generate_obs_dataformat(n_copys=self._n_copys,
                                                        item_nums=len(self.visual_dims),
                                                        name='visual')

        # process action
        ActSpace = env.action_space
        if isinstance(ActSpace, Box):
            assert len(ActSpace.shape) == 1, 'if action space is continuous, the shape length of action must equal to 1'
            self._is_continuous = True
            self.a_dim = ActSpace.shape[0]
        elif isinstance(ActSpace, Tuple):
            assert all([isinstance(i, Discrete) for i in ActSpace]) == True, 'if action space is Tuple, each item in it must have type Discrete'
            self._is_continuous = False
            self.a_dim = int(np.asarray([i.n for i in ActSpace]).prod())
        else:
            self._is_continuous = False
            self.a_dim = ActSpace.n
        env.close()

    def _partial_reset(self, obs, done):
        dones_index = np.where(done)[0]
        correct_new_obs = deepcopy(obs)
        if dones_index.shape[0] > 0:
            partial_obs = np.asarray(self._envs.reset(dones_index.tolist()))
            correct_new_obs[dones_index] = partial_obs
        return correct_new_obs
