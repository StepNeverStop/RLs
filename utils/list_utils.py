def zero_initializer(n):
    assert isinstance(n, int) and n > 0
    return [0] * n


def zeros_initializer(n, n_args):
    if n_args == 1:
        return zero_initializer(n)
    return map(zero_initializer, [n] * n_args)

def count_repeats(x, y):
    assert isinstance(x, list) and isinstance(y, list), 'assert isinstance(x, list) and isinstance(y, list)'
    assert len(x) == len(y), 'assert len(x) == len(y)'
    l = []
    for _x, _y in zip(x, y):
        [l.append(_x) for _ in range(_y)]
    return l