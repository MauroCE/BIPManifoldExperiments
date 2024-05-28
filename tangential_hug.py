import numpy as np
import time


def integrator(x0, v0, step, n_int, sign, alpha, grad_f):
    """THUG/SNUG integrator."""
    g = grad_f(x0)  # Compute gradient at x0
    g = g / np.linalg.norm(g)  # Normalize
    v0 = v0 - alpha * g * (g @ v0)  # Tilt velocity
    v, x = v0, x0  # Housekeeping

    for _ in range(n_int):
        x = x + step * v / 2  # Move to midpoint
        g = grad_f(x)  # Compute gradient at midpoint
        ghat = g / np.linalg.norm(g)  # Normalize
        v = sign*(v - 2 * (v @ ghat) * ghat)  # Reflect velocity using midpoint gradient
        x = x + step * v / 2  # Move from midpoint to end-point
    # Unsqueeze the velocity
    g = grad_f(x)
    g = g / np.linalg.norm(g)
    v = v + (alpha / (1 - alpha)) * g * (g @ v)
    return x, v


def thug(x0, step_size, B, N, alpha, log_dens, grad_f, rng=None):
    """
    Tangential Hug. Notice that it doesn't matter whether we use the gradient of pi or
    grad log pi to tilt the velocity.
    """
    start_time = time.time()
    # Grab dimension, initialize storage for samples & acceptances
    rng = np.random.default_rng(seed=np.random.randint(low=0, high=10000)) if rng is None else rng
    samples = []
    acceptances = np.zeros(N)
    esjd = 0.0  # rao-blackwellised
    for i in range(N):
        v0s = rng.normal(loc=0.0, scale=1.0, size=x0.shape)  # Draw velocity spherically
        logu = np.log(rng.uniform())  # Accept/reject
        x, v = integrator(x0=x0, v0=v0s, step=step_size, n_int=B, sign=1.0, alpha=alpha, grad_f=grad_f)
        # In the acceptance ratio must use spherical velocities, hence v0s and the unsqueezed v
        log_ar = log_dens(x) - 0.5*(v@v) - log_dens(x0) + 0.5*(v0s@v0s)
        ap = np.exp(np.clip(log_ar, a_min=None, a_max=0.0))  # acceptance probability
        esjd += ap*np.linalg.norm(x0 - x)**2
        if logu <= log_ar:
            acceptances[i] = 1         # Accepted!
            x0 = x
        samples.append(x0)
    return np.vstack(samples), acceptances.mean(), esjd, time.time() - start_time


def thug_and_snug(x0, step_thug, step_snug, p_thug, B, N, alpha, log_dens, grad_f, rng=None):
    """
    Tangential Hug. Notice that it doesn't matter whether we use the gradient of pi or
    grad log pi to tilt the velocity.
    """
    start_time = time.time()
    # Grab dimension, initialize storage for samples & acceptances
    rng = np.random.default_rng(seed=np.random.randint(low=0, high=10000)) if rng is None else rng
    samples = []
    acceptances = np.zeros(N)
    esjd = 0.0  # rao-blackwellised
    signs = rng.choice(a=[1.0, -1.0], size=N, replace=True, p=[p_thug, 1-p_thug])
    for i in range(N):
        step_size = step_thug if signs[i] == 1 else step_snug
        v0s = rng.normal(loc=0.0, scale=1.0, size=x0.shape)  # Draw velocity spherically
        logu = np.log(rng.uniform())  # Accept/reject
        x, v = integrator(x0=x0, v0=v0s, step=step_size, n_int=B, sign=signs[i], alpha=alpha, grad_f=grad_f)
        # In the acceptance ratio must use spherical velocities, hence v0s and the unsqueezed v
        log_ar = log_dens(x) - 0.5*(v@v) - log_dens(x0) + 0.5*(v0s@v0s)
        ap = np.exp(np.clip(log_ar, a_min=None, a_max=0.0))  # acceptance probability
        esjd += ap*np.linalg.norm(x0 - x)**2
        if logu <= log_ar:
            acceptances[i] = 1         # Accepted!
            x0 = x
        samples.append(x0)
    return np.vstack(samples), acceptances.mean(), esjd, time.time() - start_time


def thug_and_rwm(x0, step_thug, step_rwm, p_thug, B, N, alpha, log_dens, grad_f, rng=None):
    """
    Tangential Hug. Notice that it doesn't matter whether we use the gradient of pi or
    grad log pi to tilt the velocity.
    """
    start_time = time.time()
    # Grab dimension, initialize storage for samples & acceptances
    rng = np.random.default_rng(seed=np.random.randint(low=0, high=10000)) if rng is None else rng
    samples = []
    acceptances = np.zeros(N)
    esjd = 0.0  # rao-blackwellised
    flags = rng.binomial(n=1, p=p_thug, size=N)  # 1 = THUG, 0 = RWM
    for i in range(N):
        v0s = rng.normal(loc=0.0, scale=1.0, size=x0.shape)  # Draw velocity spherically
        logu = np.log(rng.uniform())  # Accept/reject
        if flags[i] == 1:
            x, v = integrator(x0=x0, v0=v0s, step=step_thug, n_int=B, sign=1.0, alpha=alpha, grad_f=grad_f)
        else:
            x, v = x0 + step_rwm * v0s, -v0s
        # In the acceptance ratio must use spherical velocities, hence v0s and the unsqueezed v
        log_ar = log_dens(x) - 0.5*(v@v) - log_dens(x0) + 0.5*(v0s@v0s)
        ap = np.exp(np.clip(log_ar, a_min=None, a_max=0.0))  # acceptance probability
        esjd += ap*np.linalg.norm(x0 - x)**2
        if logu <= log_ar:
            acceptances[i] = 1         # Accepted!
            x0 = x
        samples.append(x0)
    return np.vstack(samples), acceptances.mean(), esjd, time.time() - start_time
