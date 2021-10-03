import numpy as np
from gym.spaces import Box, Discrete

from rls.utils.np_utils import get_discrete_action_list


class BasicWrapper:
    def __init__(self, env):
        self.env = env
        self._is_continuous = {}
        self._mu, self._sigma, self._discrete_action_list = {}, {}, {}
        for k, action_space in env.action_spaces.items():
            if isinstance(action_space, Box):
                self._is_continuous[k] = True
                self._mu[k] = (action_space.high + action_space.low) / 2
                self._sigma[k] = (action_space.high - action_space.low) / 2
            else:
                self._is_continuous[k] = False
                _is_tuple = not isinstance(action_space, Discrete)
                if _is_tuple:
                    discrete_action_dim_list = [i.n for i in action_space]
                else:
                    discrete_action_dim_list = [action_space.n]
                self._discrete_action_list[k] = get_discrete_action_list(
                    discrete_action_dim_list)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def action_sample(self):
        actions = {}
        for k, v in self.env.action_spaces.items():
            actions[k] = v.sample()
            if self._is_continuous[k]:
                actions[k] = (actions[k] - self._mu[k]) / self._sigma[k]
        return actions

    def step(self, actions):
        return self.env.step(self.action(actions))

    def action(self, actions):
        for k, v in actions.items():
            if self.env.aec_env.dones[k]:  # TODO
                actions[k] = None
            else:
                if self._is_continuous[k]:
                    actions[k] = self._sigma[k] * v + self._mu[k]
                else:
                    actions[k] = self._discrete_action_list[k][v]
                    if isinstance(actions[k], np.ndarray):
                        actions[k] = actions[k].tolist()
        return actions
