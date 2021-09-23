#!/usr/bin/env python3
# encoding: utf-8

from rls.algorithms.register import get_model_info, register

# logo: font-size: 12, foreground character: 'O', font: 幼圆
# http://life.chacuo.net/convertfont2char

register(
    name='cql_dqn',
    path='single.offline.cql_dqn',
    is_multi=False,
    class_name='CQL_DQN',
    logo='''
　　　　　　　　　　　　　　　　　ＯＯＯ　　　　　　　　ＯＯＯ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ＯＯＯ　　　　　　　　　　　　　　　　
　　　　ＯＯＯＯＯＯ　　　　　　ＯＯＯＯＯ　　　　　　　ＯＯＯ　　　　　　　　　　　　　　　　　　　ＯＯＯＯＯＯＯＯ　　　　　　ＯＯＯＯＯ　　　　　ＯＯＯＯ　　ＯＯＯ　
　　　ＯＯＯ　　ＯＯ　　　　　ＯＯ　　ＯＯＯ　　　　　　　Ｏ　　　　　　　　　　　　　　　　　　　　　ＯＯ　　ＯＯＯ　　　　　ＯＯ　　ＯＯＯ　　　　　ＯＯＯＯ　　Ｏ　　
　　　ＯＯ　　　　Ｏ　　　　ＯＯＯ　　　ＯＯＯ　　　　　　Ｏ　　　　　　　　ＯＯ　　Ｏ　　ＯＯ　　　　ＯＯ　　　ＯＯＯ　　　ＯＯＯ　　　ＯＯＯ　　　　　ＯＯＯＯ　Ｏ　　
　　ＯＯＯ　　　　　　　　　ＯＯＯ　　　ＯＯＯ　　　　　　Ｏ　　　　　　　ＯＯＯ　ＯＯＯ　ＯＯＯ　　　ＯＯ　　　　ＯＯ　　　ＯＯＯ　　　ＯＯＯ　　　　　Ｏ　ＯＯ　Ｏ　　
　　ＯＯＯ　　　　　　　　　ＯＯＯ　　　ＯＯＯ　　　　　　Ｏ　　　　Ｏ　　　ＯＯ　ＯＯＯ　ＯＯＯ　　　ＯＯ　　　　ＯＯ　　　ＯＯＯ　　　ＯＯＯ　　　　　Ｏ　ＯＯＯＯ　　
　　　ＯＯ　　　　　　　　　　ＯＯ　　　ＯＯ　　　　　　　Ｏ　　　ＯＯ　　　　　　　　　　　　　　　　ＯＯ　　　ＯＯＯ　　　　ＯＯ　　　ＯＯ　　　　　　Ｏ　　ＯＯＯ　　
　　　ＯＯＯＯ　ＯＯ　　　　　ＯＯＯＯＯＯＯ　　　　　　ＯＯＯＯＯＯ　　　　　　　　　　　　　　　　　ＯＯＯＯＯＯＯ　　　　　ＯＯＯＯＯＯＯ　　　　　ＯＯ　　　ＯＯ　　
　　　　ＯＯＯＯＯ　　　　　　　ＯＯＯＯＯ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ＯＯＯＯＯＯＯ　　　　　　　ＯＯＯＯＯ　　　　　ＯＯＯ　　　　Ｏ　　
　　　　　　　　　　　　　　　　　　ＯＯＯＯ　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ＯＯＯＯ　　　　
    '''
)

register(
    name='bcq',
    path='single.offline.bcq',
    is_multi=False,
    class_name='BCQ',
    logo='''
　　　　ＯＯＯＯＯＯＯ　　　　　　　　　ＯＯＯＯＯＯＯ　　　　　　　　ＯＯＯＯＯＯ　　　　
　　　　　ＯＯ　ＯＯＯＯ　　　　　　　ＯＯＯＯ　ＯＯＯ　　　　　　　ＯＯＯ　ＯＯＯＯ　　　
　　　　　ＯＯ　　ＯＯＯ　　　　　　ＯＯＯＯ　　　　Ｏ　　　　　　ＯＯＯ　　　ＯＯＯＯ　　
　　　　　ＯＯ　　ＯＯＯ　　　　　　ＯＯＯ　　　　　Ｏ　　　　　　ＯＯＯ　　　　ＯＯＯ　　
　　　　　ＯＯＯＯＯＯ　　　　　　　ＯＯＯ　　　　　　　　　　　　ＯＯ　　　　　ＯＯＯ　　
　　　　　ＯＯ　ＯＯＯＯ　　　　　　ＯＯＯ　　　　　　　　　　　　ＯＯＯ　　　　ＯＯＯ　　
　　　　　ＯＯ　　ＯＯＯ　　　　　　ＯＯＯ　　　　　　　　　　　　ＯＯＯ　　　　ＯＯＯ　　
　　　　　ＯＯ　　　ＯＯ　　　　　　　ＯＯＯ　　　　Ｏ　　　　　　ＯＯＯ　　　ＯＯＯ　　　
　　　　　ＯＯ　ＯＯＯＯ　　　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　ＯＯＯＯＯＯＯＯ　　　
　　　　ＯＯＯＯＯＯＯＯ　　　　　　　　　ＯＯＯＯＯ　　　　　　　　　ＯＯＯＯＯ　　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ＯＯＯＯ　　　　
　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　ＯＯＯ　　　　
    '''
)
