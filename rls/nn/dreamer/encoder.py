import numpy as np
import torch as t
from torch import nn
from torch.nn import functional as F

from rls.nn.activations import Act_REGISTER


class VisualEncoder(nn.Module):
    """
    Encoder to embed image observation (3, 64, 64) to vector (1024,)
    """

    def __init__(self, visual_dim, depth=32, act='relu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(visual_dim[-1], 1*depth, kernel_size=4, stride=2),
            Act_REGISTER[act](),
            nn.Conv2d(1*depth, 2*depth, kernel_size=4, stride=2),
            Act_REGISTER[act](),
            nn.Conv2d(2*depth, 4*depth, kernel_size=4, stride=2),
            Act_REGISTER[act](),
            nn.Conv2d(4*depth, 8*depth, kernel_size=4, stride=2),
            Act_REGISTER[act](),
            nn.Flatten()
        )

        with t.no_grad():
            self.h_dim = np.prod(
                self.net(t.zeros(1, visual_dim[-1], visual_dim[0], visual_dim[1])).shape[1:])

    def forward(self, obs):
        tb = obs.shape[:-3]
        # [T, B, H, W, C] => [T*B, H, W, C]
        obs = obs.view((-1,)+obs.shape[-3:])
        obs = obs.permute(0, 3, 1, 2)   # [T*B, H, W, C] => [T*B, C, H, W]
        ret = self.net(obs)  # [T*B, *]
        ret = ret.view(tb+(-1,))  # [T, B, *]
        return ret


class VectorEncoder(nn.Module):
    """
    """

    def __init__(self, vector_dim):
        super().__init__()
        self.net = nn.Identity()
        self.h_dim = vector_dim

    def forward(self, obs):
        return self.net(obs)
