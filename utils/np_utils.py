import numpy as np


def all_equal(x):
    '''
    return whether items in x all equal or not.
    '''
    return (x == x.reshape(-1)[0]).all()

def get_first_item(x):
    '''
    return the first item in numpy array.
    '''
    return x.reshape(-1)[0]

def is_inf_inside(x):
    '''
    return whether np.inf, -np.inf is in x or not.
    '''
    return np.isinf(x).any()