#!/usr/bin/env python3
# encoding: utf-8

import os
import numpy as np

from copy import deepcopy
from collections import defaultdict
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.base_env import (ActionTuple,
                                    ActionSpec)  # TODO

from rls.utils.logging_utils import get_logger
logger = get_logger(__name__)

from rls.common.yaml_ops import load_yaml
from rls.utils.np_utils import get_discrete_action_list
from rls.utils.specs import (SingleAgentEnvArgs,
                             MultiAgentEnvArgs,
                             ModelObservations,
                             SingleModelInformation)
from rls.envs.unity_wrapper.core import (ObservationWrapper,
                                         ActionWrapper)


class BasicUnityEnvironment(object):

    def __init__(self, kwargs):
        self._side_channels = self.initialize_all_side_channels(kwargs)

        env_kwargs = dict(seed=int(kwargs['env_seed']),
                          worker_id=int(kwargs['worker_id']),
                          timeout_wait=int(kwargs['timeout_wait']),
                          side_channels=list(self._side_channels.values())    # 注册所有初始化后的通讯频道
                          )
        if kwargs['file_name'] is not None:
            unity_env_dict = load_yaml('/'.join([os.getcwd(), 'rls', 'envs', 'unity_env_dict.yaml']))
            env_kwargs.update(file_name=kwargs['file_name'],
                              base_port=kwargs['port'],
                              no_graphics=not kwargs['render'],
                              additional_args=[
                '--scene', str(unity_env_dict.get(kwargs.get('env_name', '3DBall'), 'None'))
            ])
        self.env = UnityEnvironment(**env_kwargs)
        self.env.reset()
        self.initialize_environment()

    def initialize_all_side_channels(self, kwargs):
        '''
        初始化所有的通讯频道
        '''
        engine_configuration_channel = EngineConfigurationChannel()
        engine_configuration_channel.set_configuration_parameters(width=kwargs['width'],
                                                                  height=kwargs['height'],
                                                                  quality_level=kwargs['quality_level'],
                                                                  time_scale=1 if bool(kwargs.get('inference', False)) else kwargs['time_scale'],
                                                                  target_frame_rate=kwargs['target_frame_rate'],
                                                                  capture_frame_rate=kwargs['capture_frame_rate'])
        float_properties_channel = EnvironmentParametersChannel()
        for k, v in kwargs.get('initialize_config', {}).items():
            float_properties_channel.set_float_parameter(k, v)
        return dict(engine_configuration_channel=engine_configuration_channel,
                    float_properties_channel=float_properties_channel)

    def reset(self, **kwargs):
        for k, v in kwargs.get('reset_config', {}).items():
            self._side_channels['float_properties_channel'].set_float_parameter(k, v)
        self.env.reset()
        obs = self.get_obs()
        return obs if self.is_multi_agents else obs[self.first_bn]

    def step(self, actions, **kwargs):
        '''
        params: actions, type of dict or np.ndarray, if the type of actions is
                not dict, then set those actions for the first behavior controller.
        '''
        for k, v in kwargs.get('step_config', {}).items():
            self._side_channels['float_properties_channel'].set_float_parameter(k, v)

        actions = deepcopy(actions)
        if self.is_multi_agents:
            assert isinstance(actions, dict)
            for k, v in actions.items():
                if self.is_continuous[k]:
                    self.empty_actiontuples[k].add_continuous(v)
                else:
                    self.empty_actiontuples[k].add_discrete(self.discrete_action_lists[k][v])
                self.env.set_actions(k, self.empty_actiontuples[k])
        else:
            # TODO:  优化
            if self.is_continuous[self.first_bn]:
                self.empty_actiontuples[self.first_bn].add_continuous(actions)
            else:
                self.empty_actiontuples[self.first_bn].add_discrete(self.discrete_action_lists[self.first_bn][actions])
            self.env.set_actions(self.first_bn, self.empty_actiontuples[self.first_bn])

        self.env.step()
        obs = self.get_obs()
        return obs if self.is_multi_agents else obs[self.first_bn]

    def initialize_environment(self):
        '''
        初始化环境，获取必要的信息，如状态、动作维度等等
        '''

        self.behavior_names = list(self.env.behavior_specs.keys())
        self.is_multi_agents = len(self.behavior_names) > 1
        self.first_bn = self.behavior_names[0]
        self.first_fbn = self.first_bn.replace('?', '_')

        self.behavior_agents, self.behavior_ids = self._get_real_agent_numbers_and_ids()  # 得到每个环境控制几个智能体

        self.vector_idxs = defaultdict(list)
        self.vector_dims = defaultdict(list)
        self.visual_idxs = defaultdict(list)
        self.visual_sources = defaultdict(int)
        self.visual_resolutions = defaultdict(list)
        self.s_dim = defaultdict(int)
        self.a_dim = defaultdict(int)
        self.discrete_action_lists = {}
        self.is_continuous = {}
        self.discrete_branchess = {}
        self.empty_actiontuples = {}

        for bn, spec in self.env.behavior_specs.items():
            for i, shape in enumerate(spec.observation_shapes):
                if len(shape) == 1:
                    self.vector_idxs[bn].append(i)
                    self.vector_dims[bn].append(shape[0])
                elif len(shape) == 3:
                    self.visual_idxs[bn].append(i)
                    self.visual_resolutions[bn].append(list(shape))  # TODO:  适配多个不同size的图像输入，目前只支持1种类型的图像输入
                else:
                    raise ValueError("shape of observation cannot be understood.")
            self.s_dim[bn] = sum(self.vector_dims[bn])
            self.visual_sources[bn] = len(self.visual_idxs[bn])

            action_spec = spec.action_spec
            if action_spec.is_continuous:
                self.a_dim[bn] = action_spec.continuous_size
                self.discrete_action_lists[bn] = None
                self.is_continuous[bn] = True
            elif action_spec.is_discrete:
                self.a_dim[bn] = int(np.asarray(action_spec.discrete_branches).prod())
                self.discrete_action_lists[bn] = get_discrete_action_list(action_spec.discrete_branches)
                self.is_continuous[bn] = False
            else:
                raise NotImplementedError("doesn't support continuous and discrete actions simultaneously for now.")

            self.empty_actiontuples[bn] = action_spec.empty_action(n_agents=self.behavior_agents[bn])

        if self.is_multi_agents:
            self.behavior_controls = defaultdict(int)
            for bn in self.behavior_names:
                self.behavior_controls[bn] = int(bn.split('#')[0])
            self.env_copys = self.behavior_agents[self.first_bn] // self.behavior_controls[self.first_bn]

    @property
    def EnvSpec(self):
        if self.is_multi_agents:
            return MultiAgentEnvArgs(
                s_dim=self.s_dim.values(),
                a_dim=self.a_dim.values(),
                visual_sources=self.visual_sources.values(),
                visual_resolutions=self.visual_resolutions.values(),
                is_continuous=self.is_continuous.values(),
                n_agents=self.behavior_agents.values(),
                behavior_controls=self.behavior_controls.values()
            )
        else:
            return SingleAgentEnvArgs(
                s_dim=self.s_dim[self.first_bn],
                a_dim=self.a_dim[self.first_bn],
                visual_sources=self.visual_sources[self.first_bn],
                visual_resolutions=self.visual_resolutions[self.first_bn],
                is_continuous=self.is_continuous[self.first_bn],
                n_agents=self.behavior_agents[self.first_bn]
            )

    def _get_real_agent_numbers_and_ids(self):
        '''获取环境中真实的智能体数量和对应的id'''
        self.env.reset()
        behavior_agents = defaultdict(int)
        behavior_ids = defaultdict(lambda: np.empty(0))
        # 10 step
        for _ in range(10):
            for bn in self.behavior_names:
                d, t = self.env.get_steps(bn)
                # TODO: 检查t是否影响
                if len(d) > len(behavior_ids[bn]):
                    behavior_agents[bn] = len(d)
                    behavior_ids[bn] = d.agent_id
                self.env.set_actions(bn, self.env.behavior_specs[bn].action_spec.random_action(n_agents=len(d)))
            self.env.step()

        for bn in self.behavior_names:
            behavior_ids[bn] = {_id: _idx for _id, _idx in zip(behavior_ids[bn], range(len(behavior_ids[bn])))}

        self.env.reset()

        return behavior_agents, behavior_ids

    def get_obs(self):
        '''
        解析环境反馈的信息，将反馈信息分为四部分：向量、图像、奖励、done信号
        '''
        rets = {}
        for bn in self.behavior_names:
            vector, visual, reward, done, corrected_vector, corrected_visual, info = self.coordinate_information(bn)
            rets[bn] = SingleModelInformation(
                corrected_obs=ModelObservations(vector=corrected_vector,
                                                visual=corrected_visual),
                obs=ModelObservations(vector=vector,
                                      visual=visual),
                reward=reward,
                done=done,
                info=info
            )
        return rets

    def coordinate_information(self, bn):
        '''
        TODO: Annotation
        '''
        n = self.behavior_agents[bn]
        ids = self.behavior_ids[bn]
        ps = []
        d, t = self.env.get_steps(bn)
        if len(t):
            ps.append(t)

        if len(d) != 0 and len(d) != n:
            raise ValueError(f'agents number error. Expected 0 or {n}, received {len(d)}')

        # some of environments done, but some of not
        while len(d) != n:
            self.env.step()
            d, t = self.env.get_steps(bn)
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

        return (self.deal_vector(n, [obs[vi] for vi in self.vector_idxs[bn]]),
                self.deal_visual(n, [obs[vi] for vi in self.visual_idxs[bn]]),
                np.asarray(reward),
                np.asarray(done),
                self.deal_vector(n, [corrected_obs[vi] for vi in self.vector_idxs[bn]]),
                self.deal_visual(n, [corrected_obs[vi] for vi in self.visual_idxs[bn]]),
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
        for bn in self.behavior_names:
            if self.is_continuous[bn]:
                actions[bn] = np.random.random((self.behavior_agents[bn], self.a_dim[bn])) * 2 - 1  # [-1, 1]
            else:
                actions[bn] = np.random.randint(self.a_dim[bn], size=(self.behavior_agents[bn],), dtype=np.int32)
        if self.is_multi_agents:
            return actions
        else:
            return actions[bn]

    def __getattr__(self, name):
        '''
        不允许获取BasicUnityEnvironment中以'_'开头的属性
        '''
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)


class ScaleVisualWrapper(ObservationWrapper):

    def observation(self, observation):

        for bn in self.behavior_names:
            observation[bn].visual = self.func(observation[bn].visual)
            observation[bn].corrected_visual = self.func(observation[bn].corrected_visual)

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
