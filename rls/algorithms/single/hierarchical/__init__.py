#!/usr/bin/env python3
# encoding: utf-8

from rls.algorithms.register import register

# logo: font-size: 12, foreground character: 'O', font: 幼圆
# http://life.chacuo.net/convertfont2char

register(
    name='aoc',
    path='single.hierarchical.aoc',
    is_multi=False,
    class_name='AOC',
    logo="""
　　　　　　　　ＯＯ　　　　　　　　　　　　　　ＯＯＯＯＯＯ　　　　　　　　　　　　　ＯＯＯＯＯＯＯ　　　　
　　　　　　　ＯＯＯ　　　　　　　　　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　　　　ＯＯＯＯＯＯＯＯＯ　　　　
　　　　　　　ＯＯＯ　　　　　　　　　　　　ＯＯＯ　　　　ＯＯＯ　　　　　　　　ＯＯＯＯ　　　　ＯＯ　　　　
　　　　　　ＯＯＯＯＯ　　　　　　　　　　ＯＯＯＯ　　　　ＯＯＯＯ　　　　　　　ＯＯＯ　　　　　　Ｏ　　　　
　　　　　　ＯＯＯＯＯ　　　　　　　　　　ＯＯＯ　　　　　　ＯＯＯ　　　　　　　ＯＯＯ　　　　　　　　　　　
　　　　　　ＯＯ　ＯＯＯ　　　　　　　　　ＯＯＯ　　　　　　ＯＯＯ　　　　　　ＯＯＯＯ　　　　　　　　　　　
　　　　　ＯＯ　　ＯＯＯ　　　　　　　　　ＯＯＯ　　　　　　ＯＯＯ　　　　　　ＯＯＯＯ　　　　　　　　　　　
　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　　ＯＯＯ　　　　　　ＯＯＯ　　　　　　　ＯＯＯ　　　　　　　　　　　
　　　　ＯＯＯ　　　ＯＯＯ　　　　　　　　ＯＯＯ　　　　　ＯＯＯＯ　　　　　　　ＯＯＯＯ　　　　　ＯＯ　　　
　　　　ＯＯ　　　　ＯＯＯ　　　　　　　　　ＯＯＯ　　　　ＯＯＯ　　　　　　　　　ＯＯＯＯ　　　ＯＯＯ　　　
　　　ＯＯＯ　　　　ＯＯＯＯ　　　　　　　　ＯＯＯＯＯＯＯＯＯ　　　　　　　　　　　ＯＯＯＯＯＯＯＯ　　　　
　　　ＯＯＯ　　　　ＯＯＯＯＯ　　　　　　　　　ＯＯＯＯＯＯ　　　　　　　　　　　　　ＯＯＯＯＯ
    """
)

register(
    name='ppoc',
    path='single.hierarchical.ppoc',
    is_multi=False,
    class_name='PPOC',
    logo="""
　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　　　　　ＯＯＯＯＯＯ　　　　　　　　　　　　　ＯＯＯＯＯＯＯ　　　　
　　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　　　　ＯＯＯＯＯＯＯＯＯ　　　　
　　　　　　　ＯＯ　　ＯＯＯＯ　　　　　　　　　　ＯＯ　　ＯＯＯＯ　　　　　　　ＯＯＯ　　　　ＯＯＯ　　　　　　　　ＯＯＯＯ　　　　ＯＯ　　　　
　　　　　　　ＯＯ　　　ＯＯＯ　　　　　　　　　　ＯＯ　　　ＯＯＯ　　　　　　ＯＯＯＯ　　　　ＯＯＯＯ　　　　　　　ＯＯＯ　　　　　　Ｏ　　　　
　　　　　　　ＯＯ　　ＯＯＯＯ　　　　　　　　　　ＯＯ　　ＯＯＯＯ　　　　　　ＯＯＯ　　　　　　ＯＯＯ　　　　　　　ＯＯＯ　　　　　　　　　　　
　　　　　　　ＯＯＯＯＯＯＯ　　　　　　　　　　　ＯＯＯＯＯＯＯ　　　　　　　ＯＯＯ　　　　　　ＯＯＯ　　　　　　ＯＯＯＯ　　　　　　　　　　　
　　　　　　　ＯＯＯＯＯＯ　　　　　　　　　　　　ＯＯＯＯＯＯ　　　　　　　　ＯＯＯ　　　　　　ＯＯＯ　　　　　　ＯＯＯＯ　　　　　　　　　　　
　　　　　　　ＯＯ　　　　　　　　　　　　　　　　ＯＯ　　　　　　　　　　　　ＯＯＯ　　　　　　ＯＯＯ　　　　　　　ＯＯＯ　　　　　　　　　　　
　　　　　　　ＯＯ　　　　　　　　　　　　　　　　ＯＯ　　　　　　　　　　　　ＯＯＯ　　　　　ＯＯＯＯ　　　　　　　ＯＯＯＯ　　　　　ＯＯ　　　
　　　　　　　ＯＯ　　　　　　　　　　　　　　　　ＯＯ　　　　　　　　　　　　　ＯＯＯ　　　　ＯＯＯ　　　　　　　　　ＯＯＯＯ　　　ＯＯＯ　　　
　　　　　ＯＯＯＯＯＯ　　　　　　　　　　　　ＯＯＯＯＯＯ　　　　　　　　　　　ＯＯＯＯＯＯＯＯＯ　　　　　　　　　　　ＯＯＯＯＯＯＯＯ　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ＯＯＯＯＯＯ　　　　　　　　　　　　　ＯＯＯＯＯ　
    """
)
register(
    name='oc',
    path='single.hierarchical.oc',
    is_multi=False,
    class_name='OC',
    logo="""
　　　　　　ＯＯＯＯＯＯ　　　　　　　　　　　　　ＯＯＯＯＯＯＯ　　　　
　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　　　　ＯＯＯＯＯＯＯＯＯ　　　　
　　　　ＯＯＯ　　　　ＯＯＯ　　　　　　　　ＯＯＯＯ　　　　ＯＯ　　　　
　　　ＯＯＯＯ　　　　ＯＯＯＯ　　　　　　　ＯＯＯ　　　　　　Ｏ　　　　
　　　ＯＯＯ　　　　　　ＯＯＯ　　　　　　　ＯＯＯ　　　　　　　　　　　
　　　ＯＯＯ　　　　　　ＯＯＯ　　　　　　ＯＯＯＯ　　　　　　　　　　　
　　　ＯＯＯ　　　　　　ＯＯＯ　　　　　　ＯＯＯＯ　　　　　　　　　　　
　　　ＯＯＯ　　　　　　ＯＯＯ　　　　　　　ＯＯＯ　　　　　　　　　　　
　　　ＯＯＯ　　　　　ＯＯＯＯ　　　　　　　ＯＯＯＯ　　　　　ＯＯ　　　
　　　　ＯＯＯ　　　　ＯＯＯ　　　　　　　　　ＯＯＯＯ　　　ＯＯＯ　　　
　　　　ＯＯＯＯＯＯＯＯＯ　　　　　　　　　　　ＯＯＯＯＯＯＯＯ　　　　
　　　　　　ＯＯＯＯＯＯ　　　　　　　　　　　　　ＯＯＯＯＯ
    """
)

register(
    name='ioc',
    path='single.hierarchical.ioc',
    is_multi=False,
    class_name='IOC',
    logo="""
　　　　　　ＯＯＯＯＯ　　　　　　　　　　　　　ＯＯＯＯＯＯ　　　　　　　　　　　　　ＯＯＯＯＯＯＯ　　　　
　　　　　　　ＯＯＯ　　　　　　　　　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　　　　ＯＯＯＯＯＯＯＯＯ　　　　
　　　　　　　　ＯＯ　　　　　　　　　　　　ＯＯＯ　　　　ＯＯＯ　　　　　　　　ＯＯＯＯ　　　　ＯＯ　　　　
　　　　　　　　ＯＯ　　　　　　　　　　　ＯＯＯＯ　　　　ＯＯＯＯ　　　　　　　ＯＯＯ　　　　　　Ｏ　　　　
　　　　　　　　ＯＯ　　　　　　　　　　　ＯＯＯ　　　　　　ＯＯＯ　　　　　　　ＯＯＯ　　　　　　　　　　　
　　　　　　　　ＯＯ　　　　　　　　　　　ＯＯＯ　　　　　　ＯＯＯ　　　　　　ＯＯＯＯ　　　　　　　　　　　
　　　　　　　　ＯＯ　　　　　　　　　　　ＯＯＯ　　　　　　ＯＯＯ　　　　　　ＯＯＯＯ　　　　　　　　　　　
　　　　　　　　ＯＯ　　　　　　　　　　　ＯＯＯ　　　　　　ＯＯＯ　　　　　　　ＯＯＯ　　　　　　　　　　　
　　　　　　　　ＯＯ　　　　　　　　　　　ＯＯＯ　　　　　ＯＯＯＯ　　　　　　　ＯＯＯＯ　　　　　ＯＯ　　　
　　　　　　　　ＯＯ　　　　　　　　　　　　ＯＯＯ　　　　ＯＯＯ　　　　　　　　　ＯＯＯＯ　　　ＯＯＯ　　　
　　　　　　ＯＯＯＯＯ　　　　　　　　　　　ＯＯＯＯＯＯＯＯＯ　　　　　　　　　　　ＯＯＯＯＯＯＯＯ　　　　
　　　　　　　　　　　　　　　　　　　　　　　　ＯＯＯＯＯＯ　　　　　　　　　　　　　ＯＯＯＯＯ
    """
)
