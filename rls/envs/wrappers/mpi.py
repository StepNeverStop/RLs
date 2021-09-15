#!/usr/bin/env python3
# encoding: utf-8
import multiprocessing
from typing import Any, Dict, List, Tuple, Union


class CloudpickleWrapper(object):
    def __init__(self, fn):
        self.fn = fn

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob):
        import pickle
        self.fn = pickle.loads(ob)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def _worker(idx, env_fn, config, conn):
    env = env_fn(idx, **config)
    try:
        while True:
            attr, data = conn.recv()
            attrs = attr.split('.')
            obj = env
            for attr in attrs:
                obj = getattr(obj, attr)
            if hasattr(obj, '__call__'):
                ret = obj(*data.get('args', ()), **data.get('kwargs', {}))
            else:
                ret = obj
            conn.send(ret)
            if attr == 'close':
                break
    except Exception as e:
        print(e)
    finally:
        env.close()


class MPIEnv:

    def __init__(self, n, env_fn, config: Dict = {}):
        # multiprocessing.set_start_method("fork")  # TODO:
        self.idxs = list(range(n))
        self.parent_conns = []
        for idx in range(n):
            parent_conn, child_conn = multiprocessing.Pipe()
            multiprocessing.Process(target=_worker,
                                    args=(idx,
                                          CloudpickleWrapper(env_fn),
                                          config,
                                          child_conn),
                                    daemon=True).start()
            self.parent_conns.append(parent_conn)

    def run(self, attr: str, params: Union[Tuple, List, Dict] = dict(args=(), kwargs=dict()), idxs=None):
        idxs = (idxs,) if isinstance(idxs, int) else idxs
        idxs = self.idxs if idxs is None else idxs
        rets = []
        if isinstance(params, dict):
            params = [params] * len(idxs)
        for i, idx in enumerate(idxs):
            self.parent_conns[idx].send((attr, params[i]))
            rets.append(self.parent_conns[idx].recv())
        return rets


if __name__ == '__main__':

    import gym

    def env_fn(idx, **config):
        _env = gym.make(**config)
        _env.seed(idx)
        return _env

    config = {'id': 'CartPole-v0'}

    env = MPIEnv(2, env_fn, config)

    print(env.run('reset'))
    print(env.run('action_space'))
    actions = env.run('action_space.sample')
    print(actions)

    params = []
    for i in range(2):
        params.append(dict(args=(actions[i],)))
    print(env.run('step', params=tuple(params)))
