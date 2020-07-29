#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Iterator


def zero_initializer(n: int) -> List[int]:
    assert isinstance(n, int) and n > 0
    return [0] * n


def zeros_initializer(n: int, n_args: int) -> Iterator:
    if n_args == 1:
        return zero_initializer(n)
    return map(zero_initializer, [n] * n_args)


def count_repeats(x: List, y: List) -> List:
    assert isinstance(x, list) and isinstance(y, list), 'assert isinstance(x, list) and isinstance(y, list)'
    assert len(x) == len(y), 'assert len(x) == len(y)'
    l = []
    for _x, _y in zip(x, y):
        [l.append(_x) for _ in range(_y)]
    return l
