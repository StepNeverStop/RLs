#!/usr/bin/env python3
# encoding: utf-8

import os
import numpy as np

from copy import deepcopy
from collections import deque
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from rls.utils.logging_utils import get_logger
from rls.utils.display import colorize
logger = get_logger(__name__)

try:
    import cv2
    cv2.ocl.setUseOpenCL(False)
except:
    logger.warning(colorize('opencv-python is needed to train visual-based model.', color='yellow'))
    pass

from rls.common.yaml_ops import load_yaml
from rls.utils.np_utils import get_discrete_action_list
from rls.utils.indexs import (SingleAgentEnvArgs,
                              MultiAgentEnvArgs)

# obs: [
#     brain1: [
#         agent1: [],
#         agent2: [],
#         ...
#         agentn: []
#     ],
#     brain2: [
#         agent1: [],
#         agent2: [],
#         ...
#         agentn: []
#     ],
#     ...
#     brainn: [
#         agent1: [],
#         agent2: [],
#         ...
#         agentn: []
#     ],
# ]

# =>

# obs: [
#     brain1: [
#         agent1: n*[],
#         agent2: n*[],
#         ...
#         agentn: n*[]
#     ],
#     brain2: [
#         agent1: n*[],
#         agent2: n*[],
#         ...
#         agentn: n*[]
#     ],
#     ...
#     brainn: [
#         agent1: n*[]
#         agent2: n*[]
#         ...
#         agentn: n*[]
#     ],
# ]


class UnityWrapper(object):

    def __init__(self, env_args):
        self.engine_configuration_channel = EngineConfigurationChannel()
        if env_args['train_mode']:
            self.engine_configuration_channel.set_configuration_parameters(time_scale=env_args['train_time_scale'])
        else:
            self.engine_configuration_channel.set_configuration_parameters(width=env_args['width'],
                                                                           height=env_args['height'],
                                                                           quality_level=env_args['quality_level'],
                                                                           time_scale=env_args['inference_time_scale'],
                                                                           target_frame_rate=env_args['target_frame_rate'])
        self.float_properties_channel = EnvironmentParametersChannel()
        if env_args['file_path'] is None:
            self._env = UnityEnvironment(base_port=5004,
                                         seed=env_args['env_seed'],
                                         side_channels=[self.engine_configuration_channel, self.float_properties_channel])
        else:
            unity_env_dict = load_yaml('/'.join([os.getcwd(), 'rls', 'envs', 'unity_env_dict.yaml']))
            self._env = UnityEnvironment(file_name=env_args['file_path'],
                                         base_port=env_args['port'],
                                         no_graphics=not env_args['render'],
                                         seed=env_args['env_seed'],
                                         side_channels=[self.engine_configuration_channel, self.float_properties_channel],
                                         additional_args=[
                                             '--scene', str(unity_env_dict.get(env_args.get('env_name', 'Roller'), 'None')),
                                             '--n_agents', str(env_args.get('env_num', 1))
            ])
        self.reset_config = env_args['reset_config']

    def reset(self, **kwargs):
        reset_config = kwargs.get('reset_config', None) or self.reset_config
        for k, v in reset_config.items():
            self.float_properties_channel.set_float_parameter(k, v)
        self._env.reset()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self._env, name)


class BasicWrapper:
    def __init__(self, env: UnityWrapper):
        self._env = env
        self._env.reset()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self._env, name)

    def process_visual_obs(self, image):
        return image


class InfoWrapper(BasicWrapper):
    def __init__(self, env, env_args):
        super().__init__(env)
        self.group_names = list(self._env.behavior_specs.keys())  # 所有脑的名字列表
        self.fixed_group_names = list(map(lambda x: x.replace('?', '_'), self.group_names))
        self.group_specs = [self._env.behavior_specs[g] for g in self.group_names]  # 所有脑的信息
        self.vector_idxs = [[i for i, g in enumerate(spec.observation_shapes) if len(g) == 1] for spec in self.group_specs]   # 得到所有脑 观测值为向量的下标
        self.vector_dims = [[g[0] for g in spec.observation_shapes if len(g) == 1] for spec in self.group_specs]  # 得到所有脑 观测值为向量的维度
        self.visual_idxs = [[i for i, g in enumerate(spec.observation_shapes) if len(g) == 3] for spec in self.group_specs]   # 得到所有脑 观测值为图像的下标
        self.group_num = len(self.group_names)

        self.visual_sources = [len(v) for v in self.visual_idxs]
        self.visual_resolutions = []
        for spec in self.group_specs:
            for g in spec.observation_shapes:
                if len(g) == 3:
                    self.visual_resolutions.append(list(g))
                    break
            else:
                self.visual_resolutions.append([])

        self.s_dim = [sum(v) for v in self.vector_dims]
        self.a_dim = [int(np.asarray(spec.action_shape).prod()) for spec in self.group_specs]
        self.discrete_action_lists = [None if spec.is_action_continuous() else get_discrete_action_list(spec.action_shape) for spec in self.group_specs]
        self.a_size = [spec.action_size for spec in self.group_specs]
        self.is_continuous = [spec.is_action_continuous() for spec in self.group_specs]

        self.group_agents, self.group_ids = self.get_real_agent_numbers_and_ids()  # 得到每个环境控制几个智能体

    def initialize(self):
        if all('#' in name for name in self.group_names):
            # use for multi-agents
            self.group_controls = list(map(lambda x: int(x.split('#')[0]), self.group_names))
            self.env_copys = self.group_agents[0] // self.group_controls[0]
            self.EnvSpec = MultiAgentEnvArgs(
                s_dim=self.s_dim,
                a_dim=self.a_dim,
                visual_sources=self.visual_sources,
                visual_resolutions=self.visual_resolutions,
                is_continuous=self.is_continuous,
                n_agents=self.group_agents,
                group_controls=self.group_controls
            )
        else:
            self.EnvSpec = [
                SingleAgentEnvArgs(
                    s_dim=self.s_dim[i],
                    a_dim=self.a_dim[i],
                    visual_sources=self.visual_sources[i],
                    visual_resolutions=self.visual_resolutions[i],
                    is_continuous=self.is_continuous[i],
                    n_agents=self.group_agents[i]
                ) for i in range(self.group_num)]

    def random_action(self):
        '''
        choose random action for each group and each agent.
        continuous: [-1, 1]
        discrete: [0-max, 0-max, ...] i.e. action dim = [2, 3] => action range from [0, 0] to [1, 2].
        '''
        actions = []
        for i in range(self.group_num):
            if self.is_continuous[i]:
                actions.append(np.random.random((self.group_agents[i], self.a_dim[i])) * 2 - 1)  # [-1, 1]
            else:
                actions.append(np.random.randint(self.a_dim[i], size=(self.group_agents[i],), dtype=np.int32))
        return actions

    def get_real_agent_numbers_and_ids(self):
        group_agents = [0] * self.group_num
        group_ids = [np.empty(0) for _ in range(self.group_num)]
        for _ in range(10):  # 10 step
            for i, gn in enumerate(self.group_names):
                d, t = self._env.get_steps(gn)
                group_agents[i] = max(group_agents[i], len(d.agent_id))
                # TODO: 检查t是否影响
                if len(d.agent_id) > len(group_ids[i]):
                    group_ids[i] = d.agent_id

                group_spec = self.group_specs[i]
                if group_spec.is_action_continuous():
                    action = np.random.randn(len(d), group_spec.action_size)
                else:
                    branch_size = group_spec.discrete_action_branches
                    action = np.column_stack([
                        np.random.randint(0, branch_size[j], size=(len(d)))
                        for j in range(len(branch_size))
                    ])
                self._env.set_actions(gn, action)
            self._env.step()

        for i in range(self.group_num):
            group_ids[i] = {_id: _idx for _id, _idx in zip(group_ids[i], range(len(group_ids[i])))}

        return group_agents, group_ids


class GrayVisualWrapper(BasicWrapper):

    def __init__(self, env):
        super().__init__(env)
        for v in self.visual_resolutions:
            if v:
                if v[-1] > 3:
                    raise Exception('visual observations have been stacked in unity environment and number > 3. You cannot sepecify gray in python.')
                v[-1] = 1

    def process_visual_obs(self, image):
        image = self._env.process_visual_obs(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image


class ResizeVisualWrapper(BasicWrapper):

    def __init__(self, env, resize=[84, 84]):
        super().__init__(env)
        self.resize = resize
        for v in self.visual_resolutions:
            if v:
                v[0], v[1] = resize[0], resize[1]

    def process_visual_obs(self, image):
        image = self._env.process_visual_obs(image)
        image = cv2.resize(image, tuple(self.resize), interpolation=cv2.INTER_AREA).reshape(list(self.resize) + [-1])
        return image


class ScaleVisualWrapper(BasicWrapper):

    def __init__(self, env):
        super().__init__(env)

    def process_visual_obs(self, image):
        image = self._env.process_visual_obs(image)
        image *= 255
        return image.astype(np.uint8)


class UnityReturnWrapper(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        self._env.reset(**kwargs)
        return self.get_obs()

    def step(self, actions):
        for k, v in actions.items():
            self._env.set_actions(k, v)
        self._env.step()
        return self.get_obs()

    def get_obs(self):
        '''
        解析环境反馈的信息，将反馈信息分为四部分：向量、图像、奖励、done信号
        '''
        vector = []
        visual = []
        reward = []
        done = []
        info = []
        corrected_vector = []
        corrected_visual = []
        for i, gn in enumerate(self.group_names):
            vec, vis, r, d, ifo, corrected_vec, corrected_vis = self.coordinate_information(i, gn)
            vector.append(vec)
            visual.append(vis)
            reward.append(r)
            done.append(d)
            info.append(ifo)
            corrected_vector.append(corrected_vec)
            corrected_visual.append(corrected_vis)
        return (vector, visual, reward, done, info, corrected_vector, corrected_visual)

    def coordinate_information(self, i, gn):
        '''
        TODO: Annotation
        '''
        n = self.group_agents[i]
        ids = self.group_ids[i]
        ps = []
        d, t = self._env.get_steps(gn)
        if len(t):
            ps.append(t)

        if len(d) != 0 and len(d) != n:
            raise ValueError(f'agents number error. Expected 0 or {n}, received {len(d)}')

        # some of environments done, but some of not
        while len(d) != n:
            self._env.step()
            d, t = self._env.get_steps(gn)
            if len(t):
                ps.append(t)

        corrected_obs, reward = d.obs, d.reward
        obs = deepcopy(corrected_obs)  # corrected_obs应包含正确的用于决策动作的下一状态
        done = np.full(n, False)
        info = dict(max_step=np.full(n, False), real_done=np.full(n, False))

        for t in ps:    # TODO: 有待优化
            _ids = np.asarray([ids[i] for i in t.agent_id], dtype=int)
            info['max_step'][_ids] = t.interrupted    # 因为达到episode最大步数而终止的
            info['real_done'][_ids[~t.interrupted]] = True  # 去掉因为max_step而done的，只记录因为失败/成功而done的
            reward[_ids] = t.reward
            done[_ids] = True
            # zip: vector, visual, ...
            for _obs, _tobs in zip(obs, t.obs):
                _obs[_ids] = _tobs

        return (self.deal_vector(n, [obs[vi] for vi in self.vector_idxs[i]]),
                self.deal_visual(n, [obs[vi] for vi in self.visual_idxs[i]]),
                np.asarray(reward),
                np.asarray(done),
                info,
                self.deal_vector(n, [corrected_obs[vi] for vi in self.vector_idxs[i]]),
                self.deal_visual(n, [corrected_obs[vi] for vi in self.visual_idxs[i]]))

    def deal_vector(self, n, vecs):
        '''
        把向量观测信息 按每个智能体 拼接起来
        '''
        if len(vecs):
            return np.hstack(vecs)
        else:
            return np.array([]).reshape(n, -1)

    def deal_visual(self, n, viss):
        '''
        viss : [camera1, camera2, camera3, ...]
        把图像观测信息 按每个智能体 组合起来
        '''
        ss = []
        for j in range(n):
            # 第j个智能体
            s = []
            for v in viss:
                s.append(self._env.process_visual_obs(v[j]))
            ss.append(np.array(s))  # [agent1(camera1, camera2, camera3, ...), ...]
        return np.array(ss)  # [B, N, (H, W, C)]


class StackVisualWrapper(BasicWrapper):

    def __init__(self, env, stack_nums=4):
        super().__init__(env)
        self._stack_nums = stack_nums
        self._stack_deque = {gn: deque([], maxlen=self._stack_nums) for gn in self.group_names}
        self._stack_deque_corrected = {gn: deque([], maxlen=self._stack_nums) for gn in self.group_names}
        for v in self.visual_resolutions:
            if v:
                if v[-1] > 3:
                    raise Exception('visual observations have been stacked in unity environment. You cannot sepecify stack in python.')
                v[-1] *= stack_nums

    def reset(self, **kwargs):
        vector, visual, reward, done, info, corrected_vector, corrected_visual = self._env.reset(**kwargs)
        for i, gn in enumerate(self.group_names):
            for _ in range(self._stack_nums):
                self._stack_deque[gn].append(visual[i])
                self._stack_deque_corrected[gn].append(corrected_visual[i])
            visual[i] = np.concatenate(self._stack_deque[gn], axis=-1)
            corrected_visual[i] = np.concatenate(self._stack_deque_corrected[gn], axis=-1)
        return (vector, visual, reward, done, info, corrected_vector, corrected_visual)

    def step(self, actions):
        vector, visual, reward, done, info, corrected_vector, corrected_visual = self._env.step(actions)
        for i, gn in enumerate(self.group_names):
            self._stack_deque[gn].append(visual[i])
            self._stack_deque_corrected[gn].append(corrected_visual[i])
        visual[i] = np.concatenate(self._stack_deque[gn], axis=-1)
        corrected_visual[i] = np.concatenate(self._stack_deque_corrected[gn], axis=-1)
        return (vector, visual, reward, done, info, corrected_vector, corrected_visual)


class ActionWrapper(BasicWrapper):

    def __init__(self, env):
        super().__init__(env)

    def step(self, actions):
        actions = deepcopy(actions)
        for i, k in enumerate(actions.keys()):
            if self.is_continuous[i]:
                pass
            else:
                actions[k] = self.discrete_action_lists[i][actions[k]]
        return self._env.step(actions)
