import numpy as np
from scipy.special import erf
from timing_utils import time_it

class WundtCurve:
    """
    Implements the Wundt curve used to compute hedonic value from novelty.
    H(x) = 2 * ( (R(x) - min_possible) / (max_possible - min_possible) ) - 1, where
      R(x) = reward(x) - alpha * punishment(x)
    reward(x) and punishment(x) are cumulative Gaussians around different means.
    """
    def __init__(self, reward_mean=0.3, reward_std=0.1,
                 punish_mean=0.7, punish_std=0.1, alpha=1):
        """
        Parameters
        ----------
        reward_mean : float
        reward_std : float
        punish_mean : float
        punish_std : float
        alpha : float
            Punishment weight
        """
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        self.punish_mean = punish_mean
        self.punish_std = punish_std
        self.alpha = alpha

    @time_it
    def cumulative_gaussian(self, x, mean, std):
        """
        F(x|mean,std) = 0.5 * [1 + erf( (x - mean) / (std * sqrt(2)) ) ]
        """
        return 0.5 * (1 + erf((x - mean)/(std * np.sqrt(2))))

    def reward(self, x):
        return self.cumulative_gaussian(x, self.reward_mean, self.reward_std)

    def punishment(self, x):
        return self.cumulative_gaussian(x, self.punish_mean, self.punish_std)

    @time_it
    def hedonic_value(self, x):
        """
        Compute hedonic value H(x) in [-1, 1].
        """
        r = self.reward(x)
        p = self.punishment(x)
        h = r - self.alpha * p
        max_possible = 1 - 0
        min_possible = 0 - self.alpha
        h_scaled = 2 * ((h - min_possible) / (max_possible - min_possible)) - 1
        return h_scaled

    def find_peak_novelty(self):
        """
        Find the novelty value that yields maximum hedonic_value in [0,1].
        """
        x_vals = np.linspace(0, 1, 1000)
        hedonic_vals = [self.hedonic_value(x) for x in x_vals]
        return x_vals[np.argmax(hedonic_vals)]
