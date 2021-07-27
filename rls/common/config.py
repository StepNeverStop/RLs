#!/usr/bin/env python3
# encoding: utf-8

from collections import defaultdict


class NamedDict(defaultdict):

    def __getattr__(self, name):
        if name in self.keys():
            return self[name]
        else:
            raise AttributeError(f'{self.__class__.__name__} don\'t have this attribute: {name}')

    def __setattr__(self, name, value):
        self[name] = value
