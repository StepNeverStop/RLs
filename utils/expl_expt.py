class ExplorationExploitationClass(object):
    """Exploration and exploitation compromise
       calculates epsilon value depending on parameters and current step number"""

    def __init__(self, eps_init=1, eps_mid=0.2, eps_final=0.01, eps_eval=0,
                 init2mid_annealing_step=500, start_step=0, max_step=5000):
        """
        From eps_init decay to eps_mid within period start_step to start_step+init2mid_annealing_step,
        Then, from eps_mid decay to eps_final within period start_step+init2mid_annealing_step to max_step.
        Args:
            eps_init: Float, Exploration probability for the first step
            eps_mid: Float, Exploration probability after
                start_step + init2mid_annealing_step steps
            eps_final: Float, Exploration probability after max_number steps
            eps_eval: Float, Exploration probability during evaluation
            init2mid_annealing_step: Int, Number of frames over which the
                exploration probabilty is annealed from eps_init to eps_mid
            start_step: Integer, Number of steps during
                which the agent only explores
            max_steps: Integer, Total number of steps
        """
        assert init2mid_annealing_step < max_step, 'init2mid_annealing_step must less than max_step.'
        self.eps_init = eps_init
        self.eps_mid = eps_mid
        self.eps_final = eps_final
        self.eps_eval = eps_eval
        self.init2mid_annealing_step = init2mid_annealing_step
        self.start_step = start_step
        self.mid_step = self.start_step + self.init2mid_annealing_step
        self.max_step = max_step

        # Slopes and intercepts for exploration decrease
        # eps_init decay to eps_mid
        self.slope_init2mid = (self.eps_mid - self.eps_init) / self.init2mid_annealing_step
        self.intercept_init2mid = self.eps_init - self.slope_init2mid * self.start_step
        # eps_mid decay to eps_final
        self.slope_mid2end = (self.eps_final - self.eps_mid) / (self.max_step - self.init2mid_annealing_step - self.start_step)
        self.intercept_mid2end = self.eps_final - self.slope_mid2end * self.max_step

    def get_esp(self, step_now, evaluation=False):
        """
        Args:
            step_now: Integer, number of the current step
        Returns:
            An integer between 0 and 1 epsilon value for the current step number
        """
        if evaluation:
            eps = self.eps_eval
        elif step_now < self.start_step:
            eps = self.eps_init
        elif self.start_step <= step_now < self.mid_step:
            eps = self.slope_init2mid * step_now + self.intercept_init2mid
        elif self.mid_step <= step_now:
            eps = self.slope_mid2end * step_now + self.intercept_mid2end
        return eps
