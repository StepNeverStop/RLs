#!/usr/bin/env python3
# encoding: utf-8

from rls.algorithms.register import get_model_info, register

# logo: font-size: 12, foreground character: 'O', font: 幼圆
# http://life.chacuo.net/convertfont2char

register(
    name='dreamer',
    path='single.modelbased.dreamer_v1',
    is_multi=False,
    class_name='DreamerV1',
    logo='''
　　　ＯＯＯＯＯＯＯＯ　　　　　　　ＯＯＯＯＯＯＯ　　　　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　　　　　ＯＯ　　　　　　　　ＯＯＯＯ　　　　ＯＯＯＯ　　　　ＯＯＯＯＯＯＯＯ　　　　　　　ＯＯＯＯＯＯＯ　　　　　
　　　　ＯＯＯＯＯＯＯＯ　　　　　　　ＯＯＯＯＯＯＯ　　　　　　　　ＯＯＯ　　ＯＯ　　　　　　　　　　ＯＯＯ　　　　　　　　　ＯＯＯ　　　　ＯＯＯ　　　　　　ＯＯＯ　　ＯＯ　　　　　　　　ＯＯＯＯＯＯＯ　　　　
　　　　ＯＯ　　　　ＯＯＯ　　　　　　ＯＯ　　ＯＯＯ　　　　　　　　ＯＯＯ　　　Ｏ　　　　　　　　　　ＯＯＯＯ　　　　　　　　ＯＯＯＯ　　ＯＯＯＯ　　　　　　ＯＯＯ　　　Ｏ　　　　　　　　ＯＯ　　ＯＯＯ　　　　
　　　　ＯＯ　　　　ＯＯＯ　　　　　　ＯＯ　　ＯＯＯ　　　　　　　　ＯＯＯ　　Ｏ　　　　　　　　　　　ＯＯＯＯ　　　　　　　　ＯＯＯＯ　　ＯＯＯＯ　　　　　　ＯＯＯ　　Ｏ　　　　　　　　　ＯＯ　　ＯＯＯ　　　　
　　　　ＯＯ　　　　　ＯＯ　　　　　　ＯＯＯＯＯＯ　　　　　　　　　ＯＯＯＯＯＯ　　　　　　　　　　ＯＯ　ＯＯ　　　　　　　　Ｏ　ＯＯ　ＯＯＯＯＯ　　　　　　ＯＯＯＯＯＯ　　　　　　　　　ＯＯＯＯＯＯ　　　　　
　　　　ＯＯ　　　　　ＯＯ　　　　　　ＯＯＯＯＯＯ　　　　　　　　　ＯＯＯ　　Ｏ　　　　　　　　　　ＯＯ　ＯＯＯ　　　　　　　Ｏ　ＯＯＯＯＯＯＯＯ　　　　　　ＯＯＯ　　Ｏ　　　　　　　　　ＯＯＯＯＯＯ　　　　　
　　　　ＯＯ　　　　ＯＯＯ　　　　　　ＯＯ　ＯＯＯＯ　　　　　　　　ＯＯＯ　　Ｏ　Ｏ　　　　　　　ＯＯＯＯＯＯＯ　　　　　　　Ｏ　　ＯＯＯ　ＯＯＯ　　　　　　ＯＯＯ　　Ｏ　Ｏ　　　　　　　ＯＯ　ＯＯＯＯ　　　　
　　　　ＯＯ　　　ＯＯＯ 　　　　　　ＯＯ　　ＯＯＯ　　　　　　　　ＯＯＯ　　　ＯＯ　　　　　　　ＯＯ　　　ＯＯＯ　　　　　　Ｏ　　ＯＯＯ　ＯＯＯ　　　　　　ＯＯＯ　　　ＯＯ　　　　　　　ＯＯ　　ＯＯＯ　　　　
　　　ＯＯＯＯＯＯＯＯ 　　　　　　ＯＯＯＯＯ　ＯＯＯＯ　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　ＯＯＯ　　　ＯＯＯ　　　　　ＯＯＯＯＯＯ　ＯＯＯＯＯ　　　　ＯＯＯＯＯＯＯＯ　　　　　　　ＯＯＯＯＯ　ＯＯＯＯ　　
    '''
)
