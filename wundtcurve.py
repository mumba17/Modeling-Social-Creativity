import numpy as np
from scipy.special import erf
from timing_utils import time_it

class WundtCurve:
    def __init__(self, reward_mean=0.3, reward_std=0.1, 
                 punish_mean=0.7, punish_std=0.1, alpha=1):
        """
        Initialize Wundt curve parameters
        reward_mean, reward_std: mean and std dev for reward Gaussian
        punish_mean, punish_std: mean and std dev for punishment Gaussian  
        alpha: punishment weight
        """
        self.reward_mean = reward_mean
        self.reward_std = reward_std
        self.punish_mean = punish_mean
        self.punish_std = punish_std
        self.alpha = alpha

    @time_it
    def cumulative_gaussian(self, x, mean, std):
        """
        Compute cumulative Gaussian as defined in paper (equation 3)
        F(x|mean,std) = 1/2[1 + erf((x-mean)/(std√2))]
        """
        return 0.5 * (1 + erf((x - mean)/(std * np.sqrt(2))))

    def reward(self, x):
        """Compute reward component R(x)"""
        return self.cumulative_gaussian(x, self.reward_mean, self.reward_std)

    def punishment(self, x):
        """Compute punishment component P(x)"""
        return self.cumulative_gaussian(x, self.punish_mean, self.punish_std)

    @time_it
    def hedonic_value(self, x):
        """
        Compute hedonic value H(x) = R(x) - αP(x)
        Returns value between -1 and 1
        """
        r = self.reward(x)
        p = self.punishment(x)
        h = r - self.alpha * p
        
        # Scale to [-1, 1] range
        max_possible = 1 - 0  # Max reward - min punishment
        min_possible = 0 - self.alpha  # Min reward - max punishment
        h_scaled = 2 * ((h - min_possible) / (max_possible - min_possible)) - 1
        
        return h_scaled

    def find_peak_novelty(self):
        """
        Find the novelty value that generates peak response (eta)
        using numerical optimization
        """
        x = np.linspace(0, 1, 1000)
        h = [self.hedonic_value(xi) for xi in x]
        return x[np.argmax(h)]
    
    def plot_curve(self):
        """ Plot the Wundt curve, showing what novelty value generates interest"""
        