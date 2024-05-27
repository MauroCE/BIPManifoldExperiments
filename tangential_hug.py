import numpy as np


def thug(x0, T, B, N, alpha, q, logpi, grad_log_pi):
    """
    Tangential Hug. Notice that it doesn't matter whether we use the gradient of pi or
    grad log pi to tilt the velocity.
    """
    # Grab dimension, initialize storage for samples & acceptances
    samples, acceptances = x0, np.zeros(N)
    esjd = 0.0  # rao-blackwellised
    for i in range(N):
        v0s = q.rvs()                    # Draw velocity spherically
        g = grad_log_pi(x0)              # Compute gradient at x0
        g = g / np.linalg.norm(g)                  # Normalize
        v0 = v0s - alpha * g * (g @ v0s)  # Tilt velocity
        v, x = v0, x0                    # Housekeeping
        logu = np.log(np.random.rand())            # Acceptance ratio
        delta = T / B                    # Compute step size

        for _ in range(B):
            x = x + delta*v/2           # Move to midpoint
            g = grad_log_pi(x)          # Compute gradient at midpoint
            ghat = g / np.linalg.norm(g)          # Normalize
            v = v - 2*(v @ ghat) * ghat  # Reflect velocity using midpoint gradient
            x = x + delta*v/2           # Move from midpoint to end-point
        # Unsqueeze the velocity
        g = grad_log_pi(x)
        g = g / np.linalg.norm(g)
        v = v + (alpha / (1 - alpha)) * g * (g @ v)
        # In the acceptance ratio must use spherical velocities, hence v0s and the unsqueezed v
        log_ar = logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0s)
        ap = np.exp(np.clip(log_ar, a_min=None, a_max=0.0))  # acceptance probability
        esjd += ap*np.linalg.norm(x0 - x)**2
        if logu <= log_ar:
            samples = np.vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
        else:
            samples = np.vstack((samples, x0))
            acceptances[i] = 0         # Rejected
    return samples[1:], acceptances, esjd
