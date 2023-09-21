#!/usr/bin/env python3
"""
Perform Bayesian optimization on a noiseless 1D Gaussian process
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Class for performing Bayesian Optimization on a 1D Gaussian process
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor
        Args:
            f: black-box function to be optimized
            X_init: numpy.ndarray shape (t, 1) representing the inputs already
                sampled with the black-box function
            Y_init: numpy.ndarray shape (t, 1) representing the outputes of the
                black-box function for each input in X_init
            t: number of initial samples
            bounds: tuple of (min, max) representing the bounds of the space in
                which to look for the optimal point
            ac_samples: the number of samples that should be analyzed during
                acquisition
            l: length parameter for the kernel
            sigma_f: the standard deviation given to the output of the
                black-box function
            xsi: the exploration-exploitation factor for acquisition
            minimize: bool determining whether optimization should be performed
                for minimization (True) or maximization (False)

        Sets the following public instance attributes:
            f: the black-box function
            gp: an instance of the class GaussianProcess
            X_s: numpy.ndarray of shape (ac_samples, 1) containing all
                acquisition sample points, evenly spaced between min and max
            xsi: the exploration-exploitation factor
            minimize: bool for minimization versus maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        start = bounds[0]
        stop = bounds[1]
        self.X_s = np.linspace(start, stop, ac_samples).reshape(ac_samples, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location using the Expected Improvement
            acquisition function

        Returns: X_next, EI
            X_next: numpy.ndarray shape (1,) representing the next best sample
                point
            EI: numpy.ndarray shape (ac_samples,) containing the expected
                improvement of each potential sample
        """
        from scipy.stats import norm

        mu, sigma = self.gp.predict(self.gp.X)
        mu_s, sigma_s = self.gp.predict(self.X_s)

        if self.minimize:
            mu_bound = np.min(mu)
        else:
            mu_bound = np.max(mu)

        Z_Numerator = mu_bound - mu_s - self.xsi
        Z = Z_Numerator / sigma_s

        EI = np.array(Z_Numerator * norm.cdf(Z) + sigma_s * norm.pdf(Z))

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI
