
from typing import Dict, List, Optional, Tuple, Union

import torch as t

from rls.nn.learningrates import LR_REGISTER
from rls.nn.modules.wrappers import TargetTwin
from rls.nn.optimizers import OP_REGISTER


class OPLR:

    def __init__(self,
                 models: Union[
                     t.Tensor,
                     t.nn.Module,
                     TargetTwin,
                     List[Union[t.Tensor, t.nn.Module, TargetTwin]],
                     Tuple[Union[t.Tensor, t.nn.Module, TargetTwin]]
                 ],
                 lr: float,
                 scheduler: str = 'default',
                 optimizer: str = 'adam',
                 *,
                 scheduler_params: Dict = {},
                 optim_params: Dict = {},
                 grad_params: Dict = {}):
        self.params = []
        if not isinstance(models, (list, tuple)):
            models = [models]

        for model in models:
            if hasattr(model, 'parameters'):
                self.params.extend(list(model.parameters()))
            else:
                self.params.append(model)

        self.optimizer = OP_REGISTER[optimizer](
            self.params, lr, **optim_params)
        self.lr_scheduler = LR_REGISTER[scheduler](
            self.optimizer, **scheduler_params)

        self._hooks = []
        if 'grad_max_norm' in grad_params.keys():
            self._hooks.append(
                lambda: t.nn.utils.clip_grad_norm_(
                    self.params, max_norm=grad_params['grad_max_norm'])
            )
        if 'grad_clip_value' in grad_params.keys():
            self._hooks.append(
                lambda: t.nn.utils.clip_grad_value_(
                    self.params, clip_value=grad_params['grad_clip_value'])
            )

    @property
    def parameters(self):
        return self.params

    @property
    def lr(self):
        return self.lr_scheduler.get_last_lr()[0]

    def zero_grad(self):
        self.optimizer.zero_grad()

    def backward(self, loss, backward_params={}):
        loss.backward(**backward_params)
        for _hook in self._hooks:
            _hook()

    def step(self):
        self.optimizer.step()
        self.lr_scheduler.step()

    def optimize(self, loss, backward_params={}):
        self.zero_grad()
        self.backward(loss, backward_params)
        self.step()

    def state_dict(self):
        return {'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
