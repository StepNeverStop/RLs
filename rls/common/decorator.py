#!/usr/bin/env python3
# encoding: utf-8

import functools
import torch as t

from rls.utils.converter import (to_numpy,
                                 to_tensor)
from rls.algorithms.base.base import Base


def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)  # 将原函数对象(func)的指定属性复制给包装函数对象(wrapper), 默认有 module、name、doc,或者通过参数选择
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper


def iTensor_oNumpy(func, dtype=t.float32, device='cpu'):

    def wrapper(*args, **kwargs):
        if args and isinstance(args[0], Base):
            device = getattr(args[0], 'device')
            args = [args[0]] + [to_tensor(x, dtype=dtype, device=device) for x in args[1:]]
        else:
            device = kwargs.pop('device') # TODO: try catch
            args = [to_tensor(x, dtype=dtype, device=device) for x in args]
        kwargs = {k: to_tensor(v, dtype=dtype, device=device) for k, v in kwargs.items()}
        output = func(*args, **kwargs)
        output = to_numpy(output)
        return output

    return wrapper
