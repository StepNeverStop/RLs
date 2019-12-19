import numpy as np
import pandas as pd
from Algorithms.tf2algos.base.policy import Policy
from utils.sth import sth


class On_Policy(Policy):
    def __init__(self,
                 s_dim,
                 visual_sources,
                 visual_resolution,
                 a_dim_or_list,
                 is_continuous,
                 **kwargs):
        super().__init__(
            s_dim=s_dim,
            visual_sources=visual_sources,
            visual_resolution=visual_resolution,
            a_dim_or_list=a_dim_or_list,
            is_continuous=is_continuous,
            **kwargs)
        self.batch_size = int(kwargs.get('batch_size', 128))
        self.data = pd.DataFrame(columns=['s', 'a', 'r', 'done'])

    def set_buffer(self, buffer):
        if buffer is None:
            self.data = pd.DataFrame(columns=['s', 'a', 'r', 'done'])
        else:
            self.data = buffer

    def store_data(self, s, visual_s, a, r, s_, visual_s_, done):
        """
        for on-policy training, use this function to store <s, a, r, s_, done> into DataFrame of Pandas.
        """
        assert isinstance(a, np.ndarray), "store need action type is np.ndarray"
        assert isinstance(r, np.ndarray), "store need reward type is np.ndarray"
        assert isinstance(done, np.ndarray), "store need done type is np.ndarray"
        if not self.is_continuous:
            a = sth.action_index2one_hot(a, self.a_dim_or_list)
        self.data = self.data.append({
            's': s,
            'visual_s': visual_s,
            'a': a,
            'r': r,
            's_': s_,
            'visual_s_': visual_s_,
            'done': done
        }, ignore_index=True)

    def no_op_store(self, *args, **kwargs):
        pass

    def clear(self):
        """
        clear the DataFrame.
        """
        self.data.drop(self.data.index, inplace=True)
