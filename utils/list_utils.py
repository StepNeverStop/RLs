def zero_initializer(n):
    assert isinstance(n, int) and n > 0
    return [0] * n

def zeros_initializer(n, n_args):
    return map(zero_initializer, [n] * n_args)