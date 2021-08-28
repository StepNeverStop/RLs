#!/usr/bin/env python3
# encoding: utf-8

import torch as t
from torch.optim import lr_scheduler

LR_REGISTER = {}
LR_REGISTER['lambda'] = lr_scheduler.LambdaLR
LR_REGISTER['step'] = lr_scheduler.StepLR
LR_REGISTER['exp'] = lr_scheduler.ExponentialLR
LR_REGISTER['cos'] = lr_scheduler.CosineAnnealingLR
LR_REGISTER['lambda'] = lr_scheduler.LambdaLR


class ConsistentLR(lr_scheduler._LRScheduler):

    def get_lr(self):

        return [group['lr'] for group in self.optimizer.param_groups]


LR_REGISTER['default'] = ConsistentLR
