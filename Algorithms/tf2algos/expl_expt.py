class ExplorationExploitationClass(object):
    """Exploration and exploitation compromise
       calculates epsilon value depending on parameters and current episode number"""

    def __init__(self, eps_initial=1, eps_final=0.2, eps_final_episode=0.01, eps_evaluation=0,
                eps_annealing_episode=500, start_episode=0, max_episode=5000):
        """
        Args:
            eps_initial: Float, Exploration probability for the first episode
            eps_final: Float, Exploration probability after
                start_episode + eps_annealing_episode episodes
            eps_final_episode: Float, Exploration probability after max_number episodes
            eps_evaluation: Float, Exploration probability during evaluation
            eps_annealing_episode: Int, Number of frames over which the
                exploration probabilty is annealed from eps_initial to eps_final
            start_episode: Integer, Number of episodes during
                which the agent only explores
            max_episodes: Integer, Total number of episodes
        """
        self.eps_initial = eps_initial
        self.eps_final = eps_final
        self.eps_final_episode = eps_final_episode
        self.eps_evaluation = eps_evaluation
        self.eps_annealing_episode = eps_annealing_episode
        self.start_episode = start_episode
        self.max_episode = max_episode

        # Slopes and intercepts for exploration decrease
        self.slope = -(self.eps_initial - self.eps_final)/self.eps_annealing_episode
        self.intercept = self.eps_initial - self.slope*self.start_episode
        self.slope_2 = -(self.eps_final - self.eps_final_episode)/(self.max_episode - self.eps_annealing_episode - self.start_episode)
        self.intercept_2 = self.eps_final_episode - self.slope_2*self.max_episode

    def get_esp(self, episode_number, evaluation=False):
        """
        Args:
            episode_number: Integer, number of the current episode
        Returns:
            An integer between 0 and 1 epsilon value for the current episode number
        """
        if evaluation:
            eps = self.eps_evaluation
        elif episode_number < self.start_episode:
            eps = self.eps_initial
        elif episode_number >= self.start_episode and episode_number < self.start_episode + self.eps_annealing_episode:
            eps = self.slope*episode_number + self.intercept
        elif episode_number >= self.start_episode + self.eps_annealing_episode:
            eps = self.slope_2*episode_number + self.intercept_2
        return eps