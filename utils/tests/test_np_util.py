import sys
sys.path.append('../..')
import numpy as np
from utils.np_utils import *

def test_all_equal():
    a = np.zeros((3,3))
    assert all_equal(a) == True
    b = np.arange(12).reshape(-1,3)
    assert all_equal(b) == False
    c = np.full((3,4), 255, dtype=np.uint8)
    assert all_equal(c) == True
    print('success')

test_all_equal()