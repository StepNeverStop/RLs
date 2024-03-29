#!/usr/bin/env python3
# encoding: utf-8

from rls.algorithms.register import register

# logo: font-size: 12, foreground character: 'O', font: 幼圆
# http://life.chacuo.net/convertfont2char

register(
    name='planet',
    path='single.modelbased.planet',
    is_multi=False,
    class_name='PlaNet',
    logo="""
　　　　ＯＯＯＯＯＯＯ　　　　　　　ＯＯＯＯＯ　　　　　　　　　　　　　　ＯＯ　　　　　　　　ＯＯＯＯ　　　ＯＯＯＯ　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　ＯＯＯＯＯＯＯＯＯ　　　
　　　　　ＯＯＯ　ＯＯＯ　　　　　　　　ＯＯ　　　　　　　　　　　　　　ＯＯＯ　　　　　　　　　　ＯＯＯ　　　　Ｏ　　　　　　　ＯＯＯ　　ＯＯ　　　　　　　ＯＯ　　ＯＯ　ＯＯ　　　
　　　　　ＯＯＯ　　ＯＯ　　　　　　　　ＯＯ　　　　　　　　　　　　　　ＯＯＯＯ　　　　　　　　　ＯＯＯＯ　　　Ｏ　　　　　　　ＯＯＯ　　　Ｏ　　　　　　　Ｏ　　　ＯＯ　　Ｏ　　　
　　　　　ＯＯＯ　ＯＯＯ　　　　　　　　ＯＯ　　　　　　　　　　　　　　ＯＯＯＯ　　　　　　　　　ＯＯＯＯＯ　　Ｏ　　　　　　　ＯＯＯ　　Ｏ　　　　　　　　　　　　ＯＯ　　　　　　
　　　　　ＯＯＯＯＯＯＯ　　　　　　　　ＯＯ　　　　　　　　　　　　　ＯＯ　ＯＯ　　　　　　　　　Ｏ　ＯＯＯＯ　Ｏ　　　　　　　ＯＯＯＯＯＯ　　　　　　　　　　　　ＯＯ　　　　　　
　　　　　ＯＯＯＯ　　　　　　　　　　　ＯＯ　　　　　　　　　　　　　ＯＯ　ＯＯＯ　　　　　　　　Ｏ　　ＯＯＯＯＯ　　　　　　　ＯＯＯ　　Ｏ　　　　　　　　　　　　ＯＯ　　　　　　
　　　　　ＯＯＯ　　　　　　　　　　　　ＯＯ　　　　Ｏ　　　　　　　ＯＯＯＯＯＯＯ　　　　　　　　Ｏ　　　ＯＯＯＯ　　　　　　　ＯＯＯ　　Ｏ　Ｏ　　　　　　　　　　ＯＯ　　　　　　
　　　　　ＯＯＯ　　　　　　　　　　　　ＯＯ　　　ＯＯ　　　　　　　ＯＯ　　　ＯＯＯ　　　　　　　Ｏ　　　　ＯＯＯ　　　　　　　ＯＯＯ　　　ＯＯ　　　　　　　　　　ＯＯ　　　　　　
　　　　ＯＯＯＯＯ　　　　　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　ＯＯＯ　　　ＯＯＯ　　　　　　ＯＯＯ　　　　ＯＯ　　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　　　ＯＯＯＯＯ　　　　　
    """
)

register(
    name='dreamer',
    path='single.modelbased.dreamer_v1',
    is_multi=False,
    class_name='DreamerV1',
    logo="""
　　　ＯＯＯＯＯＯＯＯ　　　　　　　ＯＯＯＯＯＯＯ　　　　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　　　　　ＯＯ　　　　　　　　ＯＯＯＯ　　　　ＯＯＯＯ　　　　ＯＯＯＯＯＯＯＯ　　　　　　　ＯＯＯＯＯＯＯ　　　　　
　　　　ＯＯＯＯＯＯＯＯ　　　　　　　ＯＯＯＯＯＯＯ　　　　　　　　ＯＯＯ　　ＯＯ　　　　　　　　　　ＯＯＯ　　　　　　　　　ＯＯＯ　　　　ＯＯＯ　　　　　　ＯＯＯ　　ＯＯ　　　　　　　　ＯＯＯＯＯＯＯ　　　　
　　　　ＯＯ　　　　ＯＯＯ　　　　　　ＯＯ　　ＯＯＯ　　　　　　　　ＯＯＯ　　　Ｏ　　　　　　　　　　ＯＯＯＯ　　　　　　　　ＯＯＯＯ　　ＯＯＯＯ　　　　　　ＯＯＯ　　　Ｏ　　　　　　　　ＯＯ　　ＯＯＯ　　　　
　　　　ＯＯ　　　　ＯＯＯ　　　　　　ＯＯ　　ＯＯＯ　　　　　　　　ＯＯＯ　　Ｏ　　　　　　　　　　　ＯＯＯＯ　　　　　　　　ＯＯＯＯ　　ＯＯＯＯ　　　　　　ＯＯＯ　　Ｏ　　　　　　　　　ＯＯ　　ＯＯＯ　　　　
　　　　ＯＯ　　　　　ＯＯ　　　　　　ＯＯＯＯＯＯ　　　　　　　　　ＯＯＯＯＯＯ　　　　　　　　　　ＯＯ　ＯＯ　　　　　　　　Ｏ　ＯＯ　ＯＯＯＯＯ　　　　　　ＯＯＯＯＯＯ　　　　　　　　　ＯＯＯＯＯＯ　　　　　
　　　　ＯＯ　　　　　ＯＯ　　　　　　ＯＯＯＯＯＯ　　　　　　　　　ＯＯＯ　　Ｏ　　　　　　　　　　ＯＯ　ＯＯＯ　　　　　　　Ｏ　ＯＯＯＯＯＯＯＯ　　　　　　ＯＯＯ　　Ｏ　　　　　　　　　ＯＯＯＯＯＯ　　　　　
　　　　ＯＯ　　　　ＯＯＯ　　　　　　ＯＯ　ＯＯＯＯ　　　　　　　　ＯＯＯ　　Ｏ　Ｏ　　　　　　　ＯＯＯＯＯＯＯ　　　　　　　Ｏ　　ＯＯＯ　ＯＯＯ　　　　　　ＯＯＯ　　Ｏ　Ｏ　　　　　　　ＯＯ　ＯＯＯＯ　　　　
　　　　ＯＯ　　　ＯＯＯ 　　　　　　ＯＯ　　ＯＯＯ　　　　　　　　ＯＯＯ　　　ＯＯ　　　　　　　ＯＯ　　　ＯＯＯ　　　　　　Ｏ　　ＯＯＯ　ＯＯＯ　　　　　　ＯＯＯ　　　ＯＯ　　　　　　　ＯＯ　　ＯＯＯ　　　　
　　　ＯＯＯＯＯＯＯＯ 　　　　　　ＯＯＯＯＯ　ＯＯＯＯ　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　ＯＯＯ　　　ＯＯＯ　　　　　ＯＯＯＯＯＯ　ＯＯＯＯＯ　　　　ＯＯＯＯＯＯＯＯ　　　　　　　ＯＯＯＯＯ　ＯＯＯＯ　　
    """
)

register(
    name='dreamerv2',
    path='single.modelbased.dreamer_v2',
    is_multi=False,
    class_name='DreamerV2',
    logo="""
　　　ＯＯＯＯＯＯＯＯＯＯ　　　　　　　　　ＯＯＯＯＯ　　　ＯＯＯＯ　　　　　　　　　ＯＯＯＯ　　　　　　　
　　　　ＯＯＯＯＯＯＯＯＯＯ　　　　　　　　ＯＯＯＯＯ　　　ＯＯＯＯ　　　　　　　　ＯＯＯＯＯＯ　　　　　　
　　　　　ＯＯ　　　　ＯＯＯＯ　　　　　　　　ＯＯＯ　　　　ＯＯ　　　　　　　　　　ＯＯ　ＯＯＯ　　　　　　
　　　　　ＯＯ　　　　　ＯＯＯＯ　　　　　　　　ＯＯＯ　　　ＯＯ　　　　　　　　　　Ｏ　　　ＯＯ　　　　　　
　　　　　ＯＯ　　　　　ＯＯＯＯ　　　　　　　　ＯＯＯ　　ＯＯＯ　　　　　　　　　　　　　　ＯＯ　　　　　　
　　　　　ＯＯ　　　　　　ＯＯＯ　　　　　　　　ＯＯＯ　　ＯＯ　　　　　　　　　　　　　　　ＯＯ　　　　　　
　　　　　ＯＯ　　　　　　ＯＯＯ　　　　　　　　　ＯＯＯＯＯＯ　　　　　　　　　　　　　　ＯＯ　　　　　　　
　　　　　ＯＯ　　　　　ＯＯＯＯ　　　　　　　　　ＯＯＯＯＯ　　　　　　　　　　　　　　ＯＯＯ　　　　　　　
　　　　　ＯＯ　　　　　ＯＯＯ　　　　　　　　　　　ＯＯＯＯ　　　　　　　　　　　　　ＯＯＯ　　　　　　　　
　　　　　ＯＯ　　　ＯＯＯＯＯ　　　　　　　　　　　ＯＯＯ　　　　　　　　　　　　　ＯＯＯ　　ＯＯ　　　　　
　　　ＯＯＯＯＯＯＯＯＯＯＯ　   Dreamer-v2　　　　ＯＯ　　　　　　　　　　　　ＯＯＯＯＯＯＯＯ　　　　　
    """
)

register(
    name='mve',
    path='single.modelbased.mve',
    is_multi=False,
    class_name='MVE',
    logo="""
　　ＯＯＯＯ　　　　　　ＯＯＯＯ　　　　　　ＯＯＯＯＯ　　　ＯＯＯＯ　　　　　ＯＯＯＯＯＯＯＯＯＯ　　　　　
　　　ＯＯＯＯ　　　　　ＯＯＯＯ　　　　　　ＯＯＯＯＯ　　　ＯＯＯＯ　　　　　　ＯＯＯ　　　ＯＯＯ　　　　　
　　　　ＯＯＯ　　　　ＯＯＯＯ　　　　　　　　ＯＯＯ　　　　ＯＯ　　　　　　　　　ＯＯ　　　　ＯＯ　　　　　
　　　　ＯＯＯＯ　　　ＯＯＯＯ　　　　　　　　　ＯＯＯ　　　ＯＯ　　　　　　　　　ＯＯ　　　　　Ｏ　　　　　
　　　　ＯＯＯＯ　　ＯＯＯＯＯ　　　　　　　　　ＯＯＯ　　ＯＯＯ　　　　　　　　　ＯＯ　　　ＯＯ　　　　　　
　　　　Ｏ　ＯＯＯ　ＯＯ　ＯＯ　　　　　　　　　ＯＯＯ　　ＯＯ　　　　　　　　　　ＯＯＯＯＯＯＯ　　　　　　
　　　　Ｏ　ＯＯＯＯＯＯ　ＯＯ　　　　　　　　　　ＯＯＯＯＯＯ　　　　　　　　　　ＯＯ　　　ＯＯ　　　　　　
　　　　Ｏ　ＯＯＯＯＯ　　ＯＯ　　　　　　　　　　ＯＯＯＯＯ　　　　　　　　　　　ＯＯ　　　　Ｏ　　　　　　
　　　　Ｏ　　ＯＯＯＯ　　ＯＯ　　　　　　　　　　　ＯＯＯＯ　　　　　　　　　　　ＯＯ　　　　　ＯＯ　　　　
　　　　Ｏ　　ＯＯＯ　　　ＯＯ　　　　　　　　　　　ＯＯＯ　　　　　　　　　　　　ＯＯＯ　　　ＯＯ　　　　　
　　ＯＯＯＯ　　ＯＯ　ＯＯＯＯＯＯ　　　　　　　　　　ＯＯ　　　　　　　　　　ＯＯＯＯＯＯＯＯＯＯ　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　ＯＯ　
    """
)
