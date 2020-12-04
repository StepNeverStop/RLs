#!/usr/bin/env python3
# encoding: utf-8

import os
import numpy as np

from copy import deepcopy
from collections import (deque,
                         defaultdict)
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
                              MultiAgentEnvArgs,
                              UnitySingleAgentReturn)
from rls.envs.unity_wrapper.core import (BasicWrapper,
                                         ObservationWrapper,
                                         ActionWrapper)


class BasicUnityEnvironment(object):

    def __init__(self, kwargs):
        self.reset_config = kwargs['reset_config']
        self._side_channels = self.initialize_all_side_channels(kwargs)

        env_kwargs = dict(seed=int(kwargs['env_seed']),
                          side_channels=list(self._side_channels.values())    # 注册所有初始化后的通讯频道
                          )
        if kwargs['file_path'] is not None:
            unity_env_dict = load_yaml('/'.join([os.getcwd(), 'rls', 'envs', 'unity_env_dict.yaml']))
            env_kwargs.update(file_name=kwargs['file_path'],
                              base_port=kwargs['port'],
                              no_graphics=not kwargs['render'],
                              additional_args=[
                '--scene', str(unity_env_dict.get(kwargs.get('env_name', 'Roller'), 'None')),
                '--n_agents', str(kwargs.get('env_num', 1))
            ])
        self.env = UnityEnvironment(**env_kwargs)
        self.env.reset()
        self.initialize_environment()

    def initialize_all_side_channels(self, kwargs):
        '''
        初始化所有的通讯频道
        '''
        engine_configuration_channel = EngineConfigurationChannel()
        if kwargs['train_mode']:
            engine_configuration_channel.set_configuration_parameters(time_scale=kwargs['train_time_scale'])
        else:
            engine_configuration_channel.set_configuration_parameters(width=kwargs['width'],
                                                                      height=kwargs['height'],
                                                                      quality_level=kwargs['quality_level'],
                                                                      time_scale=kwargs['inference_time_scale'],
                                                                      target_frame_rate=kwargs['target_frame_rate'])
        float_properties_channel = EnvironmentParametersChannel()
        return dict(engine_configuration_channel=engine_configuration_channel,
                    float_properties_channel=float_properties_channel
                    )

    def reset(self, **kwargs):
        # 如果注册了float_properties_channel，就使用其动态调整每个episode的环境参数
        if 'float_properties_channel' in self._side_channels.keys():
            reset_config = kwargs.get('reset_config', None) or self.reset_config
            for k, v in reset_config.items():
                self.float_properties_channel.set_float_parameter(k, v)
        self.env.reset()
        return self.get_obs()

    def step(self, actions):
        for k, v in actions.items():
            self.env.set_actions(k, v)
        self.env.step()
        return self.get_obs()

    def initialize_environment(self):
        '''
        初始化环境，获取必要的信息，如状态、动作维度等等
        '''

        # 获取所有group在Unity的名称
        self.group_names = list(self.env.behavior_specs.keys())
        self.first_gn = first_gn = self.group_names[0]
        # NOTE: 为了根据group名称建立文件夹，需要替换名称中的问号符号 TODO: 优化
        self.fixed_group_names = list(map(lambda x: x.replace('?', '_'), self.group_names))
        self.first_fgn = self.fixed_group_names[0]

        self.group_num = len(self.group_names)
        self.is_multi_agents = self.group_num > 1

        self.vector_idxs = {}
        self.vector_dims = {}
        self.visual_idxs = {}
        self.visual_sources = {}
        self.visual_resolutions = {}
        self.s_dim = {}
        self.a_dim = {}
        self.discrete_action_lists = {}
        self.is_continuous = {}

        for gn, spec in self.env.behavior_specs.items():
            # 向量输入
            self.vector_idxs[gn] = [i for i, g in enumerate(spec.observation_shapes) if len(g) == 1]
            self.vector_dims[gn] = [g[0] for g in spec.observation_shapes if len(g) == 1]
            self.s_dim[gn] = sum(self.vector_dims[gn])
            # 图像输入
            self.visual_idxs[gn] = [i for i, g in enumerate(spec.observation_shapes) if len(g) == 3]
            self.visual_sources[gn] = len(self.visual_idxs[gn])
            for g in spec.observation_shapes:
                if len(g) == 3:
                    self.visual_resolutions[gn] = list(g)
                    break
            else:
                self.visual_resolutions[gn] = []
            # 动作
            self.a_dim[gn] = int(np.asarray(spec.action_shape).prod())
            self.discrete_action_lists[gn] = None if spec.is_action_continuous() else get_discrete_action_list(spec.action_shape)
            self.is_continuous[gn] = spec.is_action_continuous()

        self.group_agents, self.group_ids = self._get_real_agent_numbers_and_ids()  # 得到每个环境控制几个智能体

        if self.is_multi_agents:
            self.group_controls = {}
            for gn in self.group_names:
                self.group_controls[gn] = int(gn.split('#')[0])
            self.env_copys = self.group_agents[first_gn] // self.group_controls[first_gn]

    @property
    def EnvSpec(self):
        if self.is_multi_agents:
            return MultiAgentEnvArgs(
                s_dim=self.s_dim.values(),
                a_dim=self.a_dim.values(),
                visual_sources=self.visual_sources.values(),
                visual_resolutions=self.visual_resolutions.values(),
                is_continuous=self.is_continuous.values(),
                n_agents=self.group_agents.values(),
                group_controls=self.group_controls.values()
            )
        else:
            return SingleAgentEnvArgs(
                s_dim=self.s_dim[self.first_gn],
                a_dim=self.a_dim[self.first_gn],
                visual_sources=self.visual_sources[self.first_gn],
                visual_resolutions=self.visual_resolutions[self.first_gn],
                is_continuous=self.is_continuous[self.first_gn],
                n_agents=self.group_agents[self.first_gn]
            )

    def _get_real_agent_numbers_and_ids(self):
        '''获取环境中真实的智能体数量和对应的id'''
        self.env.reset()
        group_agents = defaultdict(int)
        group_ids = defaultdict(lambda: np.empty(0))
        # 10 step
        for _ in range(10):
            for gn in self.group_names:
                d, t = self.env.get_steps(gn)
                group_agents[gn] = max(group_agents[gn], len(d.agent_id))
                # TODO: 检查t是否影响
                if len(d.agent_id) > len(group_ids[gn]):
                    group_ids[gn] = d.agent_id

                group_spec = self.env.behavior_specs[gn]
                if group_spec.is_action_continuous():
                    action = np.random.randn(len(d), group_spec.action_size)
                else:
                    branch_size = group_spec.discrete_action_branches
                    action = np.column_stack([
                        np.random.randint(0, branch_size[j], size=(len(d)))
                        for j in range(len(branch_size))
                    ])
                self.env.set_actions(gn, action)
            self.env.step()

        for gn in self.group_names:
            group_ids[gn] = {_id: _idx for _id, _idx in zip(group_ids[gn], range(len(group_ids[gn])))}

        self.env.reset()

        return group_agents, group_ids

    def get_obs(self):
        '''
        解析环境反馈的信息，将反馈信息分为四部分：向量、图像、奖励、done信号
        '''
        rets = {}
        for gn in self.group_names:
            vector, visual, reward, done, corrected_vector, corrected_visual, info = self.coordinate_information(gn)
            rets[gn] = UnitySingleAgentReturn(
                vector=vector,
                visual=visual,
                reward=reward,
                done=done,
                corrected_vector=corrected_vector,
                corrected_visual=corrected_visual,
                info=info
            )
        return rets

    def coordinate_information(self, gn):
        '''
        TODO: Annotation
        '''
        n = self.group_agents[gn]
        ids = self.group_ids[gn]
        ps = []
        d, t = self.env.get_steps(gn)
        if len(t):
            ps.append(t)

        if len(d) != 0 and len(d) != n:
            raise ValueError(f'agents number error. Expected 0 or {n}, received {len(d)}')

        # some of environments done, but some of not
        while len(d) != n:
            self.env.step()
            d, t = self.env.get_steps(gn)
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

        return (self.deal_vector(n, [obs[vi] for vi in self.vector_idxs[gn]]),
                self.deal_visual(n, [obs[vi] for vi in self.visual_idxs[gn]]),
                np.asarray(reward),
                np.asarray(done),
                self.deal_vector(n, [corrected_obs[vi] for vi in self.vector_idxs[gn]]),
                self.deal_visual(n, [corrected_obs[vi] for vi in self.visual_idxs[gn]]),
                info)

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
                s.append(v[j])
            ss.append(np.array(s))  # [agent1(camera1, camera2, camera3, ...), ...]
        return np.array(ss)  # [B, N, (H, W, C)]

    def random_action(self):
        '''
        choose random action for each group and each agent.
        continuous: [-1, 1]
        discrete: [0-max, 0-max, ...] i.e. action dim = [2, 3] => action range from [0, 0] to [1, 2].
        '''
        actions = {}
        for gn in self.group_names:
            if self.is_continuous[gn]:
                actions[gn] = np.random.random((self.group_agents[gn], self.a_dim[gn])) * 2 - 1  # [-1, 1]
            else:
                actions[gn] = np.random.randint(self.a_dim[gn], size=(self.group_agents[gn],), dtype=np.int32)
        return actions

    def __getattr__(self, name):
        '''
        不允许获取BasicUnityEnvironment中以'_'开头的属性
        '''
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)


class GrayVisualWrapper(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        # TODO: 有待优化
        for gn in self.group_names:
            v = self.visual_resolutions[gn]
            if v:
                if v[-1] > 3:
                    raise Exception('visual observations have been stacked in unity environment and number > 3. You cannot sepecify gray in python.')
                self.visual_resolutions[gn][-1] = 1

    def observation(self, observation):

        for gn in self.group_names:
            observation[gn].visual = self.func(observation[gn].visual)
            observation[gn].corrected_visual = self.func(observation[gn].corrected_visual)

        return observation

    def func(self, vis):
        '''
        vis: [智能体数量，摄像机数量，图像剩余维度]
        '''
        agents, cameras = vis.shape[:2]
        for i in range(agents):
            for j in range(cameras):
                vis[i, j] = cv2.cvtColor(vis[i, j], cv2.COLOR_RGB2GRAY)
        return np.asarray(vis)


class ResizeVisualWrapper(ObservationWrapper):

    def __init__(self, env, resize=[84, 84]):
        super().__init__(env)
        self.resize = resize
        for gn in self.group_names:
            if self.visual_resolutions[gn]:
                self.visual_resolutions[gn][0], self.visual_resolutions[gn][1] = resize[0], resize[1]

    def observation(self, observation):

        for gn in self.group_names:
            observation[gn].visual = self.func(observation[gn].visual)
            observation[gn].corrected_visual = self.func(observation[gn].corrected_visual)

        return observation

    def func(self, vis):
        '''
        vis: [智能体数量，摄像机数量，图像剩余维度]
        '''
        agents, cameras = vis.shape[:2]
        for i in range(agents):
            for j in range(cameras):
                vis[i, j] = cv2.resize(vis[i, j], tuple(self.resize), interpolation=cv2.INTER_AREA).reshape(list(self.resize) + [-1])
        return np.asarray(vis)


class ScaleVisualWrapper(ObservationWrapper):

    def observation(self, observation):

        for gn in self.group_names:
            observation[gn].visual = self.func(observation[gn].visual)
            observation[gn].corrected_visual = self.func(observation[gn].corrected_visual)

        return observation

    def func(self, vis):
        '''
        vis: [智能体数量，摄像机数量，图像剩余维度]
        '''
        agents, cameras = vis.shape[:2]
        for i in range(agents):
            for j in range(cameras):
                vis[i, j] *= 255
        return np.asarray(vis).astype(np.uint8)


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
        rets = self.env.reset(**kwargs)
        for gn in self.group_names:
            for _ in range(self._stack_nums):
                self._stack_deque[gn].append(rets[gn].visual)
                self._stack_deque_corrected[gn].append(rets[gn].corrected_visual)
            rets[gn].visual = np.concatenate(self._stack_deque[gn], axis=-1)
            rets[gn].corrected_visual = np.concatenate(self._stack_deque_corrected[gn], axis=-1)
        return rets

    def step(self, actions):
        rets = self.env.step(actions)
        for gn in enumerate(self.group_names):
            self._stack_deque[gn].append(rets[gn].visual)
            self._stack_deque_corrected[gn].append(rets[gn].corrected_visual)
        rets[gn].visual = np.concatenate(self._stack_deque[gn], axis=-1)
        rets[gn].corrected_visual = np.concatenate(self._stack_deque_corrected[gn], axis=-1)
        return rets


class BasicActionWrapper(ActionWrapper):

    def __init__(self, env):
        super().__init__(env)

    def action(self, actions):
        actions = deepcopy(actions)
        for gn in self.group_names:
            if not self.is_continuous[gn]:
                actions[gn] = self.discrete_action_lists[gn][actions[gn]]
        return actions
