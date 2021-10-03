#!/usr/bin/env python3
# encoding: utf-8

from typing import Dict, List, Tuple, Union


class VECEnv:

    def __init__(self, n, env_fn, config: Dict = {}):
        self.idxs = list(range(n))
        self._envs = [env_fn(idx, **config) for idx in self.idxs]

    def run(self, attr: str, params: Union[Tuple, List, Dict] = dict(args=(), kwargs=dict()), idxs=None):
        idxs = (idxs,) if isinstance(idxs, int) else idxs
        idxs = self.idxs if idxs is None else idxs
        rets = []

        if isinstance(params, dict):
            params = [params] * len(idxs)

        for i, idx in enumerate(idxs):
            attr, data = attr, params[i]
            attrs = attr.split('.')
            obj = self._envs[idx]
            for attr in attrs:
                obj = getattr(obj, attr)
            if hasattr(obj, '__call__'):
                ret = obj(*data.get('args', ()), **data.get('kwargs', {}))
            else:
                ret = obj
            rets.append(ret)

        return rets
