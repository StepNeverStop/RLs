
import numpy as np

from typing import (List,
                    NoReturn)
from gym.spaces import (Box,
                        Discrete,
                        Tuple)
from copy import deepcopy

from rls.envs.env_base import EnvBase
from rls.utils.np_utils import get_discrete_action_list
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
                 env_copys,
                 render_mode='first',
                 vector_env_type='vector',
                 seed=42,
                 **kwargs):
        '''
        Input:
            env_copys: environment number
            render_mode: mode of rendering, optional: first, last, all, random_[num] -> i.e. random_2, [list] -> i.e. [0, 2, 4]
        '''
        self._n_copys = env_copys   # environments number
        self.render_index = self._get_render_index(render_mode)
        self._initialize(env=make_env(**kwargs))
        self.envs = get_vectorized_env_class(vector_env_type)(make_env, kwargs, self._n_copys, seed)

    def reset(self, **kwargs) -> List[ModelObservations]:
        obs = np.asarray(self.envs.reset())
        if self.obs_type == 'visual':
            obs = obs[:, np.newaxis, ...]

        return [ModelObservations(vector=self.vector_info_type(*(obs,)),
                                  visual=self.visual_info_type(*(obs,)))]

    def step(self, actions: List[np.ndarray], **kwargs) -> List[SingleModelInformation]:
        actions = np.array(actions[0])  # choose the first agents' actions
        if not self._is_continuous:
            actions = self.discrete_action_list[actions]
            if self.action_type == 'discrete':
                actions = actions.reshape(-1,)
            elif self.action_type == 'Tuple(Discrete)':
                actions = actions.reshape(self._n_copys, -1).tolist()

        results = self.envs.step(actions)

        obs, reward, done, info = [np.asarray(e) for e in zip(*results)]
        reward = reward.astype('float32')
        dones_index = np.where(done)[0]
        if dones_index.shape[0] > 0:
            correct_new_obs = self._partial_reset(obs, dones_index)
        else:
            correct_new_obs = obs
        if self.obs_type == 'visual':
            obs = obs[:, np.newaxis, ...]
            correct_new_obs = correct_new_obs[:, np.newaxis, ...]

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
        self.envs.close()

    def random_action(self, **kwargs) -> List[np.ndarray]:
        '''
        generate random actions for all training environment.
        '''
        return [np.asarray(self.envs.sample())]

    def render(self, **kwargs) -> NoReturn:
        '''
        render game windows.
        '''
        record = kwargs.get('record', False)
        self.envs.render(record, self.render_index)

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
        return len(self.GroupsSpec) > 1

    # --- custom

    def _get_render_index(self, render_mode):
        '''
        get render windows list, i.e. [0, 1] when there are 4 training enviornment.
        '''
        assert isinstance(render_mode, (list, str)), 'render_mode must have type of str or list.'
        if isinstance(render_mode, list):
            assert all([isinstance(i, int) for i in render_mode]), 'items in render list must have type of int'
            assert min(index) >= 0, 'index must larger than zero'
            assert max(index) <= self._n_copys, 'render index cannot larger than environment number.'
            render_index = render_mode
        elif isinstance(render_mode, str):
            if render_mode == 'first':
                render_index = [0]
            elif render_mode == 'last':
                render_index = [-1]
            elif render_mode == 'all':
                render_index = [i for i in range(self._n_copys)]
            else:
                a, b = render_mode.split('_')
                if a == 'random' and 0 < int(b) <= self._n_copys:
                    import random
                    render_index = random.sample([i for i in range(self._n_copys)], int(b))
        else:
            raise Exception('render_mode must be first, last, all, [list] or random_[num]')
        return render_index

    def _initialize(self, env):
        assert isinstance(env.observation_space, (Box, Discrete)) and isinstance(env.action_space, (Box, Discrete)), 'action_space and observation_space must be one of available_type'
        # process observation
        ObsSpace = env.observation_space
        if isinstance(ObsSpace, Box):
            self.vector_dims = [ObsSpace.shape[0] if len(ObsSpace.shape) == 1 else 0]
            # self.obs_high = ObsSpace.high
            # self.obs_low = ObsSpace.low
        else:
            self.vector_dims = [int(ObsSpace.n)]
        if len(ObsSpace.shape) == 3:
            self.obs_type = 'visual'
            self.visual_dims = [list(ObsSpace.shape)]
        else:
            self.obs_type = 'vector'
            self.visual_dims = []

        self.vector_info_type = generate_obs_dataformat(n_copys=self._n_copys,
                                                        item_nums=1 if self.obs_type == 'vector' else 0,
                                                        name='vector')
        self.visual_info_type = generate_obs_dataformat(n_copys=self._n_copys,
                                                        item_nums=1 if self.obs_type == 'visual' else 0,
                                                        name='visual')

        # process action
        ActSpace = env.action_space
        if isinstance(ActSpace, Box):
            assert len(ActSpace.shape) == 1, 'if action space is continuous, the shape length of action must equal to 1'
            self.action_type = 'continuous'
            self._is_continuous = True
            self.a_dim = ActSpace.shape[0]
        elif isinstance(ActSpace, Tuple):
            assert all([isinstance(i, Discrete) for i in ActSpace]) == True, 'if action space is Tuple, each item in it must have type Discrete'
            self.action_type = 'Tuple(Discrete)'
            self._is_continuous = False
            self.a_dim = int(np.asarray([i.n for i in ActSpace]).prod())
            discrete_action_dim_list = [i.n for i in ActSpace]
        else:
            self.action_type = 'discrete'
            self._is_continuous = False
            self.a_dim = env.action_space.n
            discrete_action_dim_list = [env.action_space.n]
        if not self._is_continuous:
            self.discrete_action_list = get_discrete_action_list(discrete_action_dim_list)

        self.reward_threshold = env.env.spec.reward_threshold  # reward threshold refer to solved

        env.close()

    def _partial_reset(self, obs, dones_index):
        correct_new_obs = deepcopy(obs)
        partial_obs = np.asarray(self.envs.reset(dones_index.tolist()))
        correct_new_obs[dones_index] = partial_obs
        return correct_new_obs
