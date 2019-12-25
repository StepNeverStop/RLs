import functools


def lazy_property(func):
    attribute = '_lazy_' + func.__name__

    @property
    @functools.wraps(func)  # 将原函数对象(func)的指定属性复制给包装函数对象(wrapper), 默认有 module、name、doc,或者通过参数选择
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, func(self))
        return getattr(self, attribute)
    return wrapper
