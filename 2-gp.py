#!/usr/bin/env python3
"""
Create a class that represents a noiseless 1D Gaussian process
"""
import numpy as np


class GaussianProcess:
    """
    Gaussian Process Class
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class Constructor
        Args:
            X_init - numpy.ndarray shape (t, 1) representing the inputs already
                sampled with the black-box function
            Y_init - numpy.ndarray shape (t, 1) representing the outputs of the
                black-box function for each input in X_init
            l - Length parameter for the kernel
            sigma_f - Standard deviation given to the output of the black-box
                function
            Sets the public instance attribute K, representing the current
                covariance kernel matrix for the Gaussian process
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        Args:
            X1 - numpy.ndarray shape (m, 1)
            X2 - numpy.ndarray shape (n, 1)
        The kernel should use the Radiabl Basis Function (RBF)
        Returns:
            The covariance kernel matrix as a numpy.ndarray shape (m, n)
        """
        arg1 = np.sum(X1**2, 1).reshape(-1, 1)
        arg2 = np.sum(X2**2, 1)
        arg3 = np.dot(X1, X2.T)
        sqdist = arg1 + arg2 - 2 * arg3
        return self.sigma_f**2 * np.exp(-.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and std deviation of points in a Gaussian Process
        Args:
            X_s: numpy.ndarray shape (s, 1) containing all of the points whose
                mean and standard deviation should be calculated
                s: number of sample points

        Returns mu, sigma:
            mu = K_s.T * K^-1 * f
            mu: numpy.ndarray shape (s,) containing the mean for each point in
                X_s respectively
            sigma = -K_s.T * K^-1 * K_s - K_ss
            sigma: numpy.ndarray shape (s,) containing the variance for each
                point in X_s respectively
        """
        s = X_s.shape[0]

        K_s = self.kernel(self.X, X_s)
        mu = K_s.T @ np.linalg.inv(self.K) @ self.Y
        mu = mu.reshape(s,)

        K_ss = self.kernel(X_s, X_s)
        cov_s = K_ss - K_s.T @ np.linalg.inv(self.K) @ K_s
        sigma = np.diag(cov_s)

        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Method that updates a Gaussian Process
        Args:
            X_new: numpy.ndarray shape (1,) that represents a new sample point
            Y_new: numpy.ndarray shape (1,) that represents a new sample value
        Updates the public instance attributes X, Y, K
        """
        self.X = np.append(self.X, X_new)
        self.Y = np.append(self.Y, Y_new)
        self.X = self.X[:, np.newaxis]
        self.Y = self.Y[:, np.newaxis]
        self.K = self.kernel(self.X, self.X)
