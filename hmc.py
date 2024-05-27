import numpy as np
import time


def leapfrog(x, v, gnld, step, n_leapfrogs):
    """Leapfrog integrator, assuming an identity mass matrix.

    Parameters
    ----------
    :param x: Current position
    :type x: numpy.ndarray
    :param v: Current momentum/velocity
    :type v: numpy.ndarray
    :param gnld: Gradient of the negative log density
    :type gnld: function
    :param step: Step size for the integrator
    :type step: float
    :param n_leapfrogs: Number of leapfrog steps
    :type n_leapfrogs: int
    """
    assert isinstance(step, float), "Step size must be a float."
    assert isinstance(n_leapfrogs, int), "Number of leapfrog steps must be an integer."
    assert n_leapfrogs > 0, "Number of leapfrog steps must be positive."
    assert x.shape == v.shape, "Position and velocity must have the same shape."

    # first half-step for momentum
    v = v - 0.5*step*gnld(x)

    # n_leapfrog - 1 full steps for both position and velocity
    for _ in range(n_leapfrogs - 1):
        x = x + step*v
        v = v - step*gnld(x)

    # Final full position step and half-momentum step
    x = x + step*v
    v = v - 0.5*step*gnld(x)

    return x, -v  # reverse velocity for completeness


def hmc(x0, L, step, N, log_dens, gnld, rng=None):
    """Hamiltonian Monte Carlo with identity mass matrix."""
    start_time = time.time()
    rng = np.random.default_rng(seed=np.random.randint(low=0, high=10000)) if rng is None else rng
    samples = []
    acceptances = np.zeros(N)
    esjd = 0.0  # rao-blackwellised
    for i in range(N):
        # Sample a new velocity
        v0 = rng.normal(size=x0.shape)
        # Leapfrog
        xL, vL = leapfrog(x=x0, v=v0, gnld=gnld, step=step, n_leapfrogs=L)
        # Compute acceptance probability
        log_ar = log_dens(xL) - log_dens(x0) - 0.5*(vL@vL) + 0.5*(v0@v0)
        ap = np.exp(np.clip(log_ar, a_min=None, a_max=0.0))
        esjd += ap*np.linalg.norm(x0 - xL)**2
        if np.log(rng.uniform(low=0.0, high=1.0)) <= log_ar:
            acceptances[i] = 1
        samples.append(x0)
    return np.vstack(samples), acceptances, esjd, time.time() - start_time
