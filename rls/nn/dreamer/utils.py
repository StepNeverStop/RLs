from typing import Iterable

import torch as t
import torch.nn as nn


class FreezeParameters:
    def __init__(self, params: Iterable[nn.Parameter]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.params = params
        self.param_states = [p.requires_grad for p in self.params]

    def __enter__(self):
        for param in self.params:
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(self.params):
            param.requires_grad = self.param_states[i]


def compute_return(reward: t.Tensor,
                   value: t.Tensor,
                   discount: t.Tensor,
                   bootstrap: t.Tensor,
                   lambda_: float):
    """
    Compute the discounted reward for a batch of data.
    reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
    Bootstrap is [batch, 1]
    """
    next_values = t.cat([value[1:], bootstrap[None]], 0)
    target = reward + discount * next_values * (1 - lambda_)
    timesteps = list(range(reward.shape[0] - 1, -1, -1))
    outputs = []
    accumulated_reward = bootstrap
    for _t in timesteps:
        inp = target[_t]
        discount_factor = discount[_t]
        accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
        outputs.append(accumulated_reward)
    returns = t.flip(t.stack(outputs), [0])
    return returns
