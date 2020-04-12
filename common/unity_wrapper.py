
import numpy as np
import cv2
cv2.ocl.setUseOpenCL(False)
from copy import deepcopy
from utils.sth import sth
from utils.sampler import create_sampler_manager
from mlagents.mlagents_envs.environment import UnityEnvironment
from mlagents.mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from mlagents.mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel

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
        self.float_properties_channel = FloatPropertiesChannel()
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

        self.brain_names = self._env.get_agent_groups()  #所有脑的名字列表
        self.fixed_brain_names = list(map(lambda x: x.replace('?','_'), self.brain_names))
        self.brain_specs = [self._env.get_agent_group_spec(b) for b in self.brain_names] # 所有脑的信息
        self.vector_idxs = [[i for i,b in enumerate(spec.observation_shapes) if len(b)==1] for spec in self.brain_specs]   # 得到所有脑 观测值为向量的下标
        self.vector_dims = [[b[0] for b in spec.observation_shapes if len(b)==1] for spec in self.brain_specs]  # 得到所有脑 观测值为向量的维度
        self.visual_idxs = [[i for i,b in enumerate(spec.observation_shapes) if len(b)==3] for spec in self.brain_specs]   # 得到所有脑 观测值为图像的下标
        self.brain_num = len(self.brain_names)

        self.visual_sources = [len(v) for v in self.visual_idxs]
        self.visual_resolutions = []
        for spec in self.brain_specs:
            for b in spec.observation_shapes:
                if len(b) == 3:
                    self.visual_resolutions.append(
                        list(self.resize)+[list(b)[-1]])
                    break
            else:
                self.visual_resolutions.append([])

        self.s_dim = [sum(v) for v in self.vector_dims]
        self.a_dim_or_list = [spec.action_shape for spec in self.brain_specs]
        self.a_size = [spec.action_size for spec in self.brain_specs]
        self.is_continuous = [spec.is_action_continuous() for spec in self.brain_specs]

        self.brain_agents = [sr.n_agents() for sr in [self._env.get_step_result(bn) for bn in self.brain_names]]    # 得到每个环境控制几个智能体

    def random_action(self):
        '''
        choose random action for each brain and each agent.
        continuous: [-1, 1]
        discrete: [0-max, 0-max, ...] i.e. action dim = [2, 3] => action range from [0, 0] to [1, 2].
        '''
        actions = []
        for i in range(self.brain_num):
            if self.is_continuous[i]:
                actions.append(
                    np.random.random((self.brain_agents[i], self.a_dim_or_list[i])) * 2 - 1 # [-1, 1]
                )
            else:
                actions.append(
                    np.random.randint(self.a_dim_or_list[i], size=(self.brain_agents[i], self.a_size[i]), dtype=np.int32)
                )
        return actions


class UnityReturnWrapper(BasicWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._action_offset = None  # 用于记录”当前智能体信息多于初始环境智能体数量“时，需要额外拼接动作的智能体数量

    def list2dict(self, l):
        return dict([ll,idx] for idx, ll in zip(range(len(l)), l))  # agent_id : obs_idx

    def reset(self, **kwargs):
        self._env.reset(**kwargs)
        self._action_offset = {bn:0 for bn in self.brain_names}
        self._agent_ids = [self.list2dict(sr.agent_id) for sr in [self._env.get_step_result(bn) for bn in self.brain_names]]
        return self.get_obs()

    def step(self, actions):
        for k, v in actions.items():
            v = np.row_stack([v[:self._action_offset[k]], v])
            self._action_offset[k] = 0
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
            step_result = self._env.get_step_result(bn)
            vec, vis, r, d = self.coordinate_information(i, bn, step_result)
            vector.append(vec)
            visual.append(vis)
            reward.append(r)
            done.append(d)
        return zip(vector, visual, reward, done)

    def coordinate_information(self, i, bn, sr):
        '''
        处理ML-Agents从0.14.0开始返回的信息维度不一致问题。在新版ML-agents当中，如果多个环境智能体在不同的间隔中done，那么只会返回done掉的环境的信息。
        而且，如果在决策间隔的最后一时刻done，那么会返回多于原来环境数量的信息，其中既包括刚done掉的环境信息，也包括立即初始化后的信息。
        这个函数仅处理一个brain的功能。
        param: i, 指定brain的索引，也就是group的索引
        param: bn, 指定brain的名字，也就是group的名字。在这个地方传入bn主要是用于给“少”传的情景发送随机动作，使智能体尽量传回来≥环境个数的信息
        param: sr, 当前brain在当前时刻的所有信息
        '''
        _nas = sr.n_agents()    # 记录当前得到的信息中智能体的个数
        _ias = self.brain_agents[i] # 取得训练环境本身存在的智能体数量
        if _nas < _ias or (_nas == _ias and not np.isin(False, sr.done)): # 如果传回来的智能体数量小于或等于初始时，那么就单独给done掉的环境发送动作，让其尽快reset，并且把刚done掉的奖励和done信号赋值给reset后的初始状态
            _data = [(sr.agent_id, sr.reward, sr.done)]
            while True:
                self._env.step()
                sr = self._env.get_step_result(bn)
                if sr.n_agents() < _ias or (sr.n_agents() == _ias and not np.isin(False, sr.done)):
                    _data.append((sr.agent_id, sr.reward, sr.done))
                else:   # 大与或者等于
                    self._action_offset[bn] = sr.n_agents() - _ias
                    _data.append((sr.agent_id[:-_ias], sr.reward[:-_ias], sr.done[:-_ias]))
                    break
            _ids, _reward, _done = map(lambda x:np.hstack(x), zip(*_data))
        else:
            self._action_offset[bn] = _nas - _ias
            _ids, _reward, _done = sr.agent_id[:-_ias], sr.reward[:-_ias], sr.done[:-_ias]
        
        __id = [local_index for local_index, _id in enumerate(_ids) if _id in self._agent_ids[i].keys()]  # 可能存在在回合一开始就立马done的情况，这样_change_idxs中会存在None，然后报错，因此，需要在这里将多余的done序号给过滤掉
        _ids = _ids[__id]
        _reward = _reward[__id]
        _done = _done[__id]

        _r, _d = sr.reward[-_ias:], sr.done[-_ias:]
        _change_idxs = [self._agent_ids[i].get(_id) for _id in _ids]    
        if len(_change_idxs) > 0:
            _r[_change_idxs], _d[_change_idxs] = _reward, _done # TODO: 这里会偶尔报错
        _ids = np.unique(_ids)  # 经过测试发现，存在当场景中只有一个智能体时，在决策间隔内同一个智能体发送两次done信号，观测相同但是reward不同，经过上述步骤处理，这里的_ids很可能出现如下形式:[12,12]，即相同的索引，为了防止下述步骤中字典的get方法出错，因此需要去除重复
        for _id in _ids:
            _cid = self._agent_ids[i].get(_id)
            self._agent_ids[i].update({sr.agent_id[-_ias:][_cid]:self._agent_ids[i].pop(_id)})
        return (self.deal_vector(_ias, [sr.obs[vi] for vi in self.vector_idxs[i]])[-_ias:],
                self.deal_visual(_ias, [sr.obs[vi] for vi in self.visual_idxs[i]])[-_ias:],
                np.asarray(_r),
                np.asarray(_d))
        

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
        return np.array(ss)

    def resize_image(self, image):
        image = image.astype(np.uint8)
        return cv2.resize(image, tuple(self.resize), interpolation=cv2.INTER_AREA).reshape(list(self.resize)+[-1])


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
