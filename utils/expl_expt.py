class ExplorationExploitationClass(object):
    """Exploration and exploitation compromise
       calculates epsilon value depending on parameters and current episode number"""

    def __init__(self, eps_init=1, eps_mid=0.2, eps_final=0.01, eps_eval=0,
                 init2mid_annealing_episode=500, start_episode=0, max_episode=5000):
        """
        From eps_init decay to eps_mid within period start_episode to start_episode+init2mid_annealing_episode,
        Then, from eps_mid decay to eps_final within period start_episode+init2mid_annealing_episode to max_episode.
        Args:
            eps_init: Float, Exploration probability for the first episode
            eps_mid: Float, Exploration probability after
                start_episode + init2mid_annealing_episode episodes
            eps_final: Float, Exploration probability after max_number episodes
            eps_eval: Float, Exploration probability during evaluation
            init2mid_annealing_episode: Int, Number of frames over which the
                exploration probabilty is annealed from eps_init to eps_mid
            start_episode: Integer, Number of episodes during
                which the agent only explores
            max_episodes: Integer, Total number of episodes
        """
        self.eps_init = eps_init
        self.eps_mid = eps_mid
        self.eps_final = eps_final
        self.eps_eval = eps_eval
        self.init2mid_annealing_episode = init2mid_annealing_episode
        self.start_episode = start_episode
        self.mid_episode = self.start_episode + self.init2mid_annealing_episode
        self.max_episode = max_episode

        # Slopes and intercepts for exploration decrease
        # eps_init decay to eps_mid
        self.slope_init2mid = (self.eps_mid - self.eps_init) / self.init2mid_annealing_episode
        self.intercept_init2mid = self.eps_init - self.slope_init2mid * self.start_episode
        # eps_mid decay to eps_final
        self.slope_mid2end = (self.eps_final - self.eps_mid) / (self.max_episode - self.init2mid_annealing_episode - self.start_episode)
        self.intercept_mid2end = self.eps_final - self.slope_mid2end * self.max_episode

    def get_esp(self, episode_now, evaluation=False):
        """
        Args:
            episode_now: Integer, number of the current episode
        Returns:
            An integer between 0 and 1 epsilon value for the current episode number
        """
        if evaluation:
            eps = self.eps_eval
        elif episode_now < self.start_episode:
            eps = self.eps_init
        elif self.start_episode <= episode_now < self.mid_episode:
            eps = self.slope_init2mid * episode_now + self.intercept_init2mid
        elif self.mid_episode <= episode_now:
            eps = self.slope_mid2end * episode_now + self.intercept_mid2end
        return eps
