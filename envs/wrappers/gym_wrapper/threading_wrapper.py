import threading
from enum import Enum


class OP(Enum):
    RESET = 0
    STEP = 1
    CLOSE = 2
    RENDER = 3
    SAMPLE = 4


class FakeMultiThread(threading.Thread):

    def __init__(self, func, args=()):
        super().__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


def init_envs(func, config, n, seed):
    envs = [func(config) for _ in range(n)]
    seeds = [seed + i for i in range(n)]  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    [env.seed(s) for env, s in zip(envs, seeds)]
    return envs


def op_func(envs, op: OP, _args=None):
    threadpool = []
    for i, env in enumerate(envs):
        if op == OP.RESET:
            th = FakeMultiThread(env.reset, args=())
        elif op == OP.STEP:
            th = FakeMultiThread(env.step, args=(_args[i],))
        elif op == OP.CLOSE:
            th = FakeMultiThread(env.close, args=())
        elif op == OP.RENDER:
            render(envs, _args)
            return
        elif op == OP.SAMPLE:
            th = FakeMultiThread(env.action_sample, args=())
        threadpool.append(th)
    for th in threadpool:
        th.start()
    for th in threadpool:
        threading.Thread.join(th)
    return [threadpool[i].get_result() for i in range(len(envs))]


def render(envs, record):
    if record:
        [env.render(filename=r'videos/{0}-{1}.mp4'.format(env.env.spec.id, i)) for i, env in enumerate(envs)]
    else:
        [env.render() for env in envs]
