from typing import List

import torch as t
import torch.nn as nn


class VDNMixer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, q_values: List, state, **kwargs):
        '''
        params:
            q_values: N * [T, B, 1]
        '''
        return sum(q_values)    # [T, B, 1]
