import torch as t

from typing import List


class VDNMixer(t.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_values: List):
        return sum(q_values)
