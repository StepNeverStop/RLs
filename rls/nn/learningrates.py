#!/usr/bin/env python3
# encoding: utf-8

import torch as t
from torch.optim import lr_scheduler

LR_REGISTER = {}
LR_REGISTER['lambda'] = lambda optimizer, max_step: lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: max(0, 1 - step / max_step))
LR_REGISTER['step'] = lr_scheduler.StepLR
LR_REGISTER['exp'] = lr_scheduler.ExponentialLR
LR_REGISTER['cos'] = lr_scheduler.CosineAnnealingLR


class ConsistentLR(lr_scheduler._LRScheduler):

    def get_lr(self):

        return [group['lr'] for group in self.optimizer.param_groups]


LR_REGISTER['default'] = ConsistentLR
