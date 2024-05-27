import numpy as np
import scipy as sp
from warnings import catch_warnings, filterwarnings


def forward(theta):
    """Forward function for the inverse problem, mapping from R^2 to R."""
    return theta[1]**2 + 3.0*(theta[0]**2)*(theta[0]**2 - 1)


def constraint(theta, y=1.0):
    """Constraint function. For these experiments we always consider observed data to be y=1."""
    return forward(theta) - y


def grad_forward(theta):
    """Gradient of the forward function."""
    return np.array([12*theta[0]**3 - 6.0*theta[0], 2*theta[1]])


def log_prior(theta):
    """Log prior for theta is a standard normal."""
    return -0.5*np.sum(theta**2)


def log_post(theta, sigma):
    """Log posterior for theta given a noise scale sigma."""
    return log_prior(theta) - 0.5*(constraint(theta)**2)/(sigma**2)


def grad_neg_log_post(theta, sigma):
    """Gradient of the negative log posterior for theta."""
    return theta + constraint(theta)*grad_forward(theta)/sigma**2


def forward_extended(xi, sigma):
    """Extended forward function containing the noise scale."""
    return forward(xi[:2]) + sigma*xi[2]


def constraint_extended(xi, sigma, y=1.0):
    """Extended constraint function."""
    return forward_extended(xi, sigma) - y


def grad_forward_extended(xi):
    """Gradient of the extended forward function."""
    return np.array([12*xi[0]**3 - 6.0*xi[0], 2*xi[1], xi[2]])


def log_prior_extended(xi):
    """Log prior for xi is a standard normal."""
    return -0.5*np.sum(xi**2)


def log_post_extended(xi):
    """Log posterior for xi given a noise scale sigma."""
    return log_prior_extended(xi) - np.log(np.linalg.norm(grad_forward_extended(xi)))


def find_point_on_theta_manifold(maxiter=1000, tol=1e-12, rng=None, y=1.0):
    """Finds a point on the theta manifold."""
    rng = np.random.default_rng(seed=np.random.randint(low=0, high=10000)) if rng is None else rng
    iteration = 0
    with catch_warnings():
        filterwarnings('error')
        while iteration <= maxiter:
            iteration += 1
            try:
                theta0 = rng.normal(loc=0.0, scale=1.0)
                theta1_sol = sp.optimize.fsolve(
                    func=lambda theta1: constraint(np.array([theta0, *theta1]), y=y),
                    x0=rng.normal(loc=0.0, scale=1.0, size=1)
                )
                theta_found = np.array([theta0, *theta1_sol])
                if abs(constraint(theta_found, y=y)) <= tol:
                    return theta_found
            except RuntimeWarning:
                continue


def find_point_on_xi_manifold(sigma, theta_fixed=None, maxiter=500, eta_max=5.0, rng=None, y=1.0):
    """Finds a point on the extended manifold."""
    rng = np.random.default_rng(seed=np.random.randint(low=0, high=10000)) if rng is None else rng
    iteration = 0
    with catch_warnings():
        filterwarnings('error')
        while iteration <= maxiter:
            iteration += 1
            try:
                theta_init = theta_fixed if theta_fixed is not None else rng.normal(loc=0.0, scale=1, size=2)
                eta = -constraint_extended(np.array([*theta_init, 0.0]), sigma=sigma, y=y) / sigma
                if abs(eta) < eta_max:
                    return np.array([*theta_init, eta])
            except RuntimeWarning:
                continue
