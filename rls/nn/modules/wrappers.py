from copy import deepcopy

import torch as t

from rls.utils.torch_utils import sync_params


class TargetTwin:

    def __init__(self, module, ployak=0.):
        self._module = module
        self._target_module = deepcopy(module)
        self._target_module.eval()
        self.ployak = ployak

        self.sync()
        self.sync()

    def __call__(self, *args, **kwargs):
        return self._module(*args, **kwargs)

    def __repr__(self):
        return str(self._module)

    def t(self, *args, **kwargs):
        return self._target_module(*args, **kwargs)

    @property
    def target(self):
        return self._target_module

    def to(self, *args, **kwargs):
        self._module = self._module.to(*args, **kwargs)
        self._target_module = self._target_module.to(*args, **kwargs)
        return self

    def sync(self):
        sync_params(self._target_module, self._module, self.ployak)

    def __getattr__(self, attr):
        # https://github.com/python-babel/flask-babel/commit/8319a7f44f4a0b97298d20ad702f7618e6bdab6a
        # https://stackoverflow.com/questions/47299243/recursionerror-when-python-copy-deepcopy
        if attr == "__setstate__":
            raise AttributeError(attr)
        if hasattr(self._module, attr):
            return getattr(self._module, attr)
        raise AttributeError(attr)
