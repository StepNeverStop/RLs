#!/usr/bin/env python3
# encoding: utf-8
import numpy as np


class EpsilonLinearDecay(object):
    """Exploration and exploitation compromise
       calculates epsilon value depending on parameters and current step number"""

    def __init__(self, eps_init=1., eps_mid=0.2, eps_final=0.01,
                 init2mid_annealing_step=500, start_step=0, max_step=5000):
        """
        From eps_init decay to eps_mid within period start_step to start_step+init2mid_annealing_step,
        Then, from eps_mid decay to eps_final within period start_step+init2mid_annealing_step to max_step.
        Args:
            eps_init: Exploration probability for the first step
            eps_mid: Exploration probability after
                start_step + init2mid_annealing_step steps
            eps_final: Exploration probability after max_number steps
            init2mid_annealing_step: Number of frames over which the exploration probabilty is annealed from eps_init to eps_mid
            start_step: Number of steps during which the agent only explores
            max_steps: Total number of steps
        """
        self._eps_init = eps_init
        self._eps_mid = eps_mid
        self._eps_final = eps_final

        self._start_step = start_step
        self._init2mid_annealing_step = init2mid_annealing_step
        self._mid_step = self._start_step + self._init2mid_annealing_step
        self._max_step = max(max_step, self._mid_step + 1)

        # eps_init decay to eps_mid
        self._intercept_init2mid = (self._eps_init - self._eps_mid) / self._init2mid_annealing_step
        # eps_mid decay to eps_final
        self._intercept_mid2end = (self._eps_mid - self._eps_final) / (self._max_step - self._mid_step)

    def get_esp(self, step_now):
        """
        Args:
            step_now: number of the current step
        Returns:
            An integer between 0 and 1 epsilon value for the current step number
        """
        if step_now < self._start_step:
            eps = self._eps_init
        elif self._start_step <= step_now <= self._mid_step:
            eps = self._eps_init - (step_now - self._start_step) * self._intercept_init2mid
        elif self._mid_step <= step_now <= self._max_step:
            eps = self._eps_mid - (step_now - self._mid_step) * self._intercept_mid2end
        else:
            eps = self._eps_final
        return eps

    def is_random(self, step_now):
        return np.random.uniform() < self.get_esp(step_now)
