import numpy as np


class Manifold:
    def __init__(self, m, d):
        """
        Generic Manifold Class.

        Parameters
        ----------
        :param m: Number of constrains and co-dimension of the manifold
        :type m: int
        :param d: Dimension of the manifold
        :type d: int
        """
        self.m = m
        self.d = d

    def tangent_basis(self, Q):
        """
        Computes a tangent basis from the Q matrix (the transpose of the Jacobian matrix).

        Parameters
        ----------
        :param Q: 2D Numpy array of dimension (m + d, m) containing gradients of the constraints as columns
        :type Q: numpy.ndarray
        :return: Matrix containing basis of tangent space as its columns
        :rtype: numpy.ndarray
        """
        assert Q.shape == (self.m + self.d, self.m), "Q must have shape ({}, {}) but found shape {}".format(
            self.m + self.d, self.m, Q.shape)
        return np.linalg.svd(Q)[0][:, self.m:]

    def get_dimension(self):
        """Returns dimension of the manifold d."""
        return self.d

    def get_codimension(self):
        """Returns co-dimension of the manifold m."""
        return self.m


class BIPManifold(Manifold):
    def __init__(self, sigma, y_star):
        """Simple 2D Bayesian Inverse Problem.

        Parameters
        ----------
        :param sigma: Standard deviation of the noise
        :type sigma: float
        :param y_star: Observations
        :type y_star: float
        """
        super().__init__(m=1, d=2)
        self.n = self.m + self.d   # Dimension of ambient space
        self.sigma = sigma
        self.y_star = y_star

    def q(self, xi):
        """Constraint for toy BIP.

        Parameters
        ----------
        :param xi: 1D Numpy array of dimension (3, ) containing the point in the ambient space
        :type xi: numpy.ndarray
        """
        return np.array([xi[1]**2 + 3 * xi[0]**2 * (xi[0]**2 - 1)]) + self.sigma*xi[2] - self.y_star

    def Q(self, xi):
        """Transpose of Jacobian for toy BIP. """
        return np.array([12*xi[0]**3 - 6*xi[0], 2*xi[1], self.sigma]).reshape(-1, self.m)

    def log_post(self, xi):
        """log posterior for c-rwm"""
        jac = self.Q(xi).T
        return - xi[:2]@xi[:2]/2 - xi[-1]**2/2 - np.log(jac@jac.T + self.sigma**2)[0, 0]/2
