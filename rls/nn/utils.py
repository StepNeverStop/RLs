
import torch as t

from typing import (Union,
                    List,
                    Tuple,
                    Optional,
                    Dict)

from rls.nn.learningrates import LR_REGISTER
from rls.nn.optimizers import OP_REGISTER
from rls.nn.modules.wrappers import TargetTwin


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
                 optimizer_params: Dict = {},
                 clipvalue: Optional[float] = None,
                 clipnorm: Optional[float] = None):
        self.params = []
        if not isinstance(models, (list, tuple)):
            models = [models]

        for model in models:
            if hasattr(model, 'parameters'):
                self.params.extend(list(model.parameters()))
            else:
                self.params.append(model)

        self.optimizer = OP_REGISTER[optimizer](
            self.params, lr, **optimizer_params)
        self.lr_scheduler = LR_REGISTER[scheduler](
            self.optimizer, **scheduler_params)

        self.clipnorm = clipnorm
        self.clipvalue = clipvalue

        self.step = self._make_step_function()

    @property
    def parameters(self):
        return self.params

    @property
    def lr(self):
        return self.lr_scheduler.get_last_lr()[0]

    def _make_step_function(self):
        # TODO: Optimization
        if self.clipnorm and self.clipvalue:
            def func(loss):
                self.optimizer.zero_grad()
                loss.backward()
                t.nn.utils.clip_grad_norm_(self.params, max_norm=self.clipnorm)
                t.nn.utils.clip_grad_value_(
                    self.params, clip_value=self.clipvalue)
                self.optimizer.step()
                self.lr_scheduler.step()
            return func
        elif self.clipnorm:
            def func(loss):
                self.optimizer.zero_grad()
                loss.backward()
                t.nn.utils.clip_grad_norm_(self.params, max_norm=self.clipnorm)
                self.optimizer.step()
                self.lr_scheduler.step()
            return func
        elif self.clipvalue:
            def func(loss):
                self.optimizer.zero_grad()
                loss.backward()
                t.nn.utils.clip_grad_value_(
                    self.params, clip_value=self.clipvalue)
                self.optimizer.step()
                self.lr_scheduler.step()
            return func
        else:
            def func(loss):
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
            return func

    def state_dict(self):
        return {'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict()}

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.lr_scheduler.load_state_dict(state_dict['lr_scheduler'])
