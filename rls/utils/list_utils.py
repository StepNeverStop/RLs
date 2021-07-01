#!/usr/bin/env python3
# encoding: utf-8

from typing import List, Iterator


def zero_initializer(n: int) -> List[int]:
    assert isinstance(n, int) and n > 0
    return [0] * n


def zeros_initializer(n: int, n_args: int) -> Iterator:
    if n_args == 1:
        return zero_initializer(n)
    return map(zero_initializer, [n] * n_args)
