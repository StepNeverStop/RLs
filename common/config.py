from copy import deepcopy


class Config(object):
    '''
    store config parameters in this class
    self.k = v
    '''

    def __init__(self, **kwargs):
        self.add_dict(kwargs)

    @property
    def to_dict(self):
        d = deepcopy(self.__dict__)
        for k, v in d.items():
            if isinstance(v, Config):
                d[k] = v.to_dict
        return d

    def add_dict(self, d):
        assert isinstance(d, dict)
        for k, v in d.items():
            if isinstance(v, dict):
                setattr(self, k, Config(**v))
                continue
            setattr(self, k, v)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                setattr(self, k, Config(**v))
                continue
            setattr(self, k, v)

    def get(self, k, default=None):
        '''
        dict.get(k, default_value)
        '''
        if k in self.__dict__:
            return getattr(self, k)
        else:
            return default

    def update(self, d):
        assert isinstance(d, dict)
        for k, v in d.items():
            if v is not None:
                setattr(self, k, v)

    def __getattr__(self, name):
        '''
        self.name
        '''
        raise AttributeError(f'{self.__class__.__name__} don\'t have this attribute: {name}')

    def __getitem__(self, x):
        '''
        dict[x]
        '''
        return getattr(self, x)

    def __setitem__(self, x, value):
        '''
        dict[x] = value
        dict[x] += value
        '''
        return setattr(self, x, value)

    def __repr__(self):
        return '{%s}' % ',\n '.join('%r: %r' % i for i in sorted(self.to_dict.items()))


if __name__ == "__main__":
    c = Config(**dict([
        ['x', 1],
        ['y', 2]
    ]))
    print(c)
    c.add_dict(dict([
        ['z', 3],
        ['a', 4]
    ]))
    print(c)
    c.add(b=4, c=4)
    print(c)
    c['x'] = 10
    print(c)
    c['x'] += 10
    print(c)
    c.update(dict([
        ['x', 'hhh'],
        ['y', None]
    ]))
    print(c)
    cc = Config(a=1, b=2)
    c.add(cc=cc)
    print(c.to_dict)
    print(c.error)  # raise error
