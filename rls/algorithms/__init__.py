#!/usr/bin/env python3
# encoding: utf-8

from rls.algorithms.multi import *
from rls.algorithms.register import get_model_info, register
from rls.algorithms.single import *

# logo: font-size: 12, foreground character: 'O', font: 幼圆
# http://life.chacuo.net/convertfont2char

# register new algorithms like this:
# register(
#     name='pg',
#     path='single.pg',
#     is_multi=False,
#     class_name='PG',
#     logo="""
#     　　　ＯＯＯＯＯＯＯＯ　　　　　　　　　　　　ＯＯＯＯＯ　Ｏ　　　　
# 　　　　　　ＯＯＯＯＯＯＯＯ　　　　　　　　　ＯＯＯＯＯＯＯＯＯ　　　　
# 　　　　　　　ＯＯ　　ＯＯＯＯ　　　　　　　ＯＯＯＯ　　　　ＯＯ　　　　
# 　　　　　　　ＯＯ　　　ＯＯＯ　　　　　　ＯＯＯＯ　　　　　　Ｏ　　　　
# 　　　　　　　ＯＯ　　ＯＯＯＯ　　　　　　ＯＯＯ　　　　　　　　　　　　
# 　　　　　　　ＯＯＯＯＯＯＯ　　　　　　　ＯＯＯ　　　　ＯＯＯＯＯＯ　　
# 　　　　　　　ＯＯＯＯＯＯ　　　　　　　　ＯＯＯ　　　　　ＯＯＯＯＯ　　
# 　　　　　　　ＯＯ　　　　　　　　　　　　ＯＯＯ　　　　　　ＯＯ　　　　
# 　　　　　　　ＯＯ　　　　　　　　　　　　ＯＯＯ　　　　　　ＯＯ　　　　
# 　　　　　　　ＯＯ　　　　　　　　　　　　　ＯＯＯ　　　　　ＯＯ　　　　
# 　　　　　ＯＯＯＯＯＯ　　　　　　　　　　　ＯＯＯＯＯ　ＯＯＯＯ　　　　
# 　　　　　　　　　　　　　　　　　　　　　　　　ＯＯＯＯＯＯＯＯ　
#     """
# )
