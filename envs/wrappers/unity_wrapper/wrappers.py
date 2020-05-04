
import cv2
cv2.ocl.setUseOpenCL(False)
import numpy as np
from copy import deepcopy
from utils.sth import sth
from utils.sampler import create_sampler_manager
from mlagents.mlagents_envs.environment import UnityEnvironment
from mlagents.mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents.mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

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
            self._env = UnityEnvironment(file_name=env_args['file_path'],
                                         base_port=env_args['port'],
                                         no_graphics=not env_args['render'],
                                         seed=env_args['env_seed'],
                                         side_channels=[self.engine_configuration_channel, self.float_properties_channel])

    def reset(self, **kwargs):
        reset_config = kwargs.get('reset_config', {})
        for k, v in reset_config.items():
            self.float_properties_channel.set_property(k, v)
        self._env.reset()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self._env, name)

class BasicWrapper:
    def __init__(self, env):
        self._env = env
        self._env.reset()

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self._env, name)


class InfoWrapper(BasicWrapper):
    def __init__(self, env, env_args):
        super().__init__(env)
        self.resize = env_args['resize']

        self.brain_names = self._env.get_behavior_names()  #所有脑的名字列表
        self.fixed_brain_names = list(map(lambda x: x.replace('?','_'), self.brain_names))
        self.brain_specs = [self._env.get_behavior_spec(b) for b in self.brain_names] # 所有脑的信息
        self.vector_idxs = [[i for i,b in enumerate(spec.observation_shapes) if len(b)==1] for spec in self.brain_specs]   # 得到所有脑 观测值为向量的下标
        self.vector_dims = [[b[0] for b in spec.observation_shapes if len(b)==1] for spec in self.brain_specs]  # 得到所有脑 观测值为向量的维度
        self.visual_idxs = [[i for i,b in enumerate(spec.observation_shapes) if len(b)==3] for spec in self.brain_specs]   # 得到所有脑 观测值为图像的下标
        self.brain_num = len(self.brain_names)

        self.visual_sources = [len(v) for v in self.visual_idxs]
        self.visual_resolutions = []
        stack_visual_nums = env_args['stack_visual_nums'] if env_args['stack_visual_nums'] > 1 else 1
        for spec in self.brain_specs:
            for b in spec.observation_shapes:
                if len(b) == 3:
                    self.visual_resolutions.append(
                        list(self.resize)+[list(b)[-1] * stack_visual_nums])
                    break
            else:
                self.visual_resolutions.append([])

        self.s_dim = [sum(v) for v in self.vector_dims]
        self.a_dim_or_list = [spec.action_shape for spec in self.brain_specs]
        self.a_size = [spec.action_size for spec in self.brain_specs]
        self.is_continuous = [spec.is_action_continuous() for spec in self.brain_specs]

        self.brain_agents = [len(d) for d in [self._env.get_steps(bn)[0] for bn in self.brain_names]]    # 得到每个环境控制几个智能体

    def random_action(self):
        '''
        choose random action for each brain and each agent.
        continuous: [-1, 1]
        discrete: [0-max, 0-max, ...] i.e. action dim = [2, 3] => action range from [0, 0] to [1, 2].
        '''
        actions = []
        for i in range(self.brain_num):
            if self.is_continuous[i]:
                actions.append(np.random.random((self.brain_agents[i], self.a_dim_or_list[i])) * 2 - 1) # [-1, 1]
            else:
                actions.append(np.random.randint(self.a_dim_or_list[i], size=(self.brain_agents[i], self.a_size[i]), dtype=np.int32))
        return actions


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
        for i, bn in enumerate(self.brain_names):
            vec, vis, r, d = self.coordinate_information(i, bn)
            vector.append(vec)
            visual.append(vis)
            reward.append(r)
            done.append(d)
        return zip(vector, visual, reward, done)

    def coordinate_information(self, i, bn):
        '''
        TODO: Annotation
        '''
        n = self.brain_agents[i]
        d, t = self._env.get_steps(bn)
        ps = [t]

        if len(d) != 0 and len(d) != n:
            raise ValueError('agents number error.')

        while len(d) != n:
            self._env.step()
            d, t = self._env.get_steps(bn)
            ps.append(t)
        
        obs, reward = d.obs, d.reward
        done = np.full(n, False)

        for t in ps:    # TODO: 有待优化
            if len(t) != 0:
                reward[t.agent_id] = t.reward
                done[t.agent_id] = True
                for _obs, _tobs in zip(obs, t.obs):
                    _obs[t.agent_id] = _tobs

        return (self.deal_vector(n, [obs[vi] for vi in self.vector_idxs[i]]),
                self.deal_visual(n, [obs[vi] for vi in self.visual_idxs[i]]),
                np.asarray(reward),
                np.asarray(done))
        

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
            s = []
            for v in viss:
                s.append(self.resize_image(v[j]))
            ss.append(np.array(s))  # [agent1(camera1, camera2, camera3, ...), ...]
        return np.array(ss) # [B, N, (H, W, C)]

    def resize_image(self, image):
        image = cv2.resize(image, tuple(self.resize), interpolation=cv2.INTER_AREA).reshape(list(self.resize)+[-1])
        return image


class SamplerWrapper(BasicWrapper):
    
    def __init__(self, env, env_args):
        super().__init__(env)
        self.reset_config = env_args['reset_config']
        self.sampler_manager, self.resample_interval = create_sampler_manager(env_args['sampler_path'], 0)
        self.episode = 0

    def reset(self):
        self.episode += 1
        if self.episode % self.resample_interval == 0:
            self.reset_config.update(self.sampler_manager.sample_all())
        obs = self._env.reset(config=self.reset_config)
        return obs

class ActionWrapper(BasicWrapper):
    
    def __init__(self, env):
        super().__init__(env)

    def step(self, actions):
        actions = deepcopy(actions)
        for i, k in enumerate(actions.keys()):
            if self.is_continuous[i]:
                pass
            else:
                actions[k] = sth.int2action_index(actions[k], self.a_dim_or_list[i])
        return self._env.step(actions)
