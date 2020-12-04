

class BasicWrapper:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError("attempted to get missing private attribute '{}'".format(name))
        return getattr(self.env, name)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, actions):
        return self.env.step(actions)


class ObservationWrapper(BasicWrapper):

    def reset(self, **kwargs):
        return self.observation(self.env.reset(**kwargs))

    def step(self, actions):
        return self.observation(self.env.step(actions))

    def observation(self, observation):
        raise NotImplementedError


class ActionWrapper(BasicWrapper):

    def step(self, actions):
        return self.env.step(self.action(actions))

    def action(self, actions):
        raise NotImplementedError
