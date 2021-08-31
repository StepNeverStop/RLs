from typing import Iterable

from torch.nn import Module, Parameter


class FreezeParameters:
    def __init__(self, params: Iterable[Parameter]):
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
