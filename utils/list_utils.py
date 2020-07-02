def zero_initializer(n):
    assert isinstance(n, int) and n > 0
    return [0] * n


def zeros_initializer(n, n_args):
    if n_args == 1:
        return zero_initializer(n)
    return map(zero_initializer, [n] * n_args)
