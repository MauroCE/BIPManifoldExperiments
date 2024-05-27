import numpy as np
import time


def thug(x0, T, B, N, alpha, logpi, grad_f, rng=None):
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
        g = grad_f(x0)              # Compute gradient at x0
        g = g / np.linalg.norm(g)                  # Normalize
        v0 = v0s - alpha * g * (g @ v0s)  # Tilt velocity
        v, x = v0, x0                    # Housekeeping
        logu = np.log(rng.uniform())            # Acceptance ratio
        delta = T / B                    # Compute step size

        for _ in range(B):
            x = x + delta*v/2           # Move to midpoint
            g = grad_f(x)          # Compute gradient at midpoint
            ghat = g / np.linalg.norm(g)          # Normalize
            v = v - 2*(v @ ghat) * ghat  # Reflect velocity using midpoint gradient
            x = x + delta*v/2           # Move from midpoint to end-point
        # Unsqueeze the velocity
        g = grad_f(x)
        g = g / np.linalg.norm(g)
        v = v + (alpha / (1 - alpha)) * g * (g @ v)
        # In the acceptance ratio must use spherical velocities, hence v0s and the unsqueezed v
        log_ar = logpi(x) - 0.5*(v@v) - logpi(x0) + 0.5*(v0s@v0s)
        ap = np.exp(np.clip(log_ar, a_min=None, a_max=0.0))  # acceptance probability
        esjd += ap*np.linalg.norm(x0 - x)**2
        if logu <= log_ar:
            acceptances[i] = 1         # Accepted!
            x0 = x
        samples.append(x0)
    return np.vstack(samples), acceptances.mean(), esjd, time.time() - start_time


def ts(x0, T, thug_step, snug_step, N, alpha, logpi, grad_f, rng=None, p_thug=0.8):
    """
    THUG+SNUG mixture Notice that it doesn't matter whether we use the gradient of pi or
    grad log pi to tilt the velocity.
    """
    start_time = time.time()
    # Grab dimension, initialize storage for samples & acceptances
    rng = np.random.default_rng(seed=np.random.randint(low=0, high=10000)) if rng is None else rng
    # Sample flags for SNUG or THUG at each iteration
    iotas = rng.choice(a=[-1.0, 1.0], size=N, replace=True, p=[1 - p_thug, p_thug])
    samples = []
    acceptances = np.zeros(N)
    esjd = 0.0  # rao-blackwellised
    for i in range(N):
        v0s = rng.normal(loc=0.0, scale=1.0, size=x0.shape)  # Draw velocity spherically
        g = grad_f(x0)              # Compute gradient at x0
        g = g / np.linalg.norm(g)                  # Normalize
        v0 = v0s - alpha * g * (g @ v0s)  # Tilt velocity
        v, x = v0, x0                    # Housekeeping
        logu = np.log(rng.uniform())            # Acceptance ratio
        step = thug_step if iotas[i] == 1.0 else snug_step
        for _ in range(T):
            x = x + step*v/2           # Move to midpoint
            g = grad_f(x)          # Compute gradient at midpoint
            ghat = g / np.linalg.norm(g)          # Normalize
            v = iotas[i]*(v - 2*(v @ ghat) * ghat)  # Reflect velocity using midpoint gradient
            x = x + step*v/2           # Move from midpoint to end-point
        # Unsqueeze the velocity
        g = grad_f(x)
        g = g / np.linalg.norm(g)
        v = v + (alpha / (1 - alpha)) * g * (g @ v)
        # In the acceptance ratio must use spherical velocities, hence v0s and the unsqueezed v
        log_ar = logpi(x) - 0.5*(v@v) - logpi(x0) + 0.5*(v0s@v0s)
        ap = np.exp(np.clip(log_ar, a_min=None, a_max=0.0))  # acceptance probability
        esjd += ap*np.linalg.norm(x0 - x)**2
        if logu <= log_ar:
            acceptances[i] = 1         # Accepted!
            x0 = x
        samples.append(x0)
    return np.vstack(samples), acceptances.mean(), esjd, time.time() - start_time
