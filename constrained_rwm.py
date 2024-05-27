import numpy as np
import scipy as sp
import time


def qr_project(v, J):
    """Projects onto the tangent space using QR decomposition."""
    Q = sp.linalg.qr(J.T, mode='economic')[0]
    return Q @ (Q.T @ v)


def constrained_rwm_step(x, v, tol, maxiter, Jx, manifold):
    """Used for both forward and backward. See Manifold-Lifting paper."""
    # Project momentum
    v_projected = v - qr_project(v, Jx)
    # Unconstrained position step
    x_unconstr = x + v_projected
    # Position Projection
    a, flag, n_grad = project_zappa_manifold(manifold, x_unconstr, Jx.T, tol, maxiter)
    y = x_unconstr - Jx.T @ a
    try:
        Jy = manifold.Q(y).T
    except ValueError as e:
        print("Jacobian computation at projected point failed. ", e)
        return x, v, Jx, 0, n_grad + 1
    # backward velocity
    v_would_have = y - x
    # Find backward momentum & project it to tangent space at new position
    v_projected_endposition = v_would_have - qr_project(v_would_have, Jy)
    # Return projected position, projected momentum and flag
    return y, v_projected_endposition, Jy, flag, n_grad + 1


def constrained_leapfrog(x0, v0, J0, B, tol, rev_tol, maxiter, manifold):
    """Constrained Leapfrog/RATTLE."""
    successful = True
    n_jacobian_evaluations = 0
    x, v, J = x0, v0, J0
    for _ in range(B):
        xf, vf, Jf, converged_fw, n_fw = constrained_rwm_step(x, v, tol, maxiter, J, manifold)
        xr, vr, Jr, converged_bw, n_bw = constrained_rwm_step(xf, -vf, tol, maxiter, Jf, manifold)
        n_jacobian_evaluations += (n_fw + n_bw)  # +2 due to the line Jy = manifold.Q(y).T
        if (not converged_fw) or (not converged_bw) or (np.linalg.norm(xr - x) >= rev_tol):
            successful = False
            return x0, v0, J0, successful, n_jacobian_evaluations
        else:
            x = xf
            v = vf
            J = Jf
    return x, v, J, successful, n_jacobian_evaluations


def project_zappa_manifold(manifold, z, Q, tol=1.48e-08, maxiter=50):
    """
    This version is the version of Miranda & Zappa. It returns i, the number of iterations
    i.e. the number of gradient evaluations used.
    """
    a, flag, i = np.zeros(Q.shape[1]), 1, 0

    # Compute the constrained at z - Q@a. If it fails due to overflow error, return a rejection altogether.
    try:
        projected_value = manifold.q(z - Q @ a)
    except ValueError:
        return a, 0, i
    # While loop
    while sp.linalg.norm(projected_value) >= tol:
        try:
            Jproj = manifold.Q(z - Q @ a).T
        except ValueError as e:
            print("Jproj failed. ", e)
            return np.zeros(Q.shape[1]), 0, i
        # Check that Jproj@Q is invertible. Do this by checking condition number
        # see https://stackoverflow.com/questions/13249108/efficient-pythonic-check-for-singular-matrix
        GramMatrix = Jproj @ Q
        if np.linalg.cond(GramMatrix) < 1 / np.finfo(z.dtype).eps:
            delta_a = sp.linalg.solve(GramMatrix, projected_value)
            a += delta_a
            i += 1
            if i > maxiter:
                return np.zeros(Q.shape[1]), 0, i
            # If we are not at maxiter iteration, compute new projected value
            try:
                projected_value = manifold.q(z - Q @ a)
            except ValueError:
                return np.zeros(Q.shape[1]), 0, i
        else:
            # Fail
            return np.zeros(Q.shape[1]), 0, i
    return a, 1, i


def crwm(x0, manifold, n, T, B, tol, rev_tol, maxiter=50, rng=None):
    """C-RWM using RATTLE."""
    start_time = time.time()
    assert isinstance(B, int), "Number of integration steps B must be an integer."
    assert isinstance(n, int), "Number of samples n must be an integer."
    assert len(x0) == manifold.n, "Initial point has wrong dimension."
    rng = np.random.default_rng(seed=np.random.randint(low=0, high=10000)) if rng is None else rng
    # Settings
    step = T / B
    d, m = manifold.get_dimension(), manifold.get_codimension()

    # Initial point on the manifold
    x = x0

    # Storage
    samples = np.zeros((n, d + m))  # Store n samples on the manifold
    samples[0, :] = x
    n_evals = {'jacobian': 0, 'density': 0}
    accepted = np.zeros(n)
    esjd = 0.0                          # rao-blackwellised expected square jump distance

    # Log-uniforms for MH accept-reject step
    logu = np.log(rng.uniform(size=n))

    # Compute jacobian & density value
    Jx = manifold.Q(x).T
    logp_x = manifold.log_post(x)
    n_evals['jacobian'] += 1
    n_evals['density'] += 1

    for i in range(n):
        v = step * rng.normal(size=(m + d))  # Sample in the ambient space.
        xp, vp, Jp, LEAPFROG_SUCCESSFUL, n_jac_evals = constrained_leapfrog(x, v, Jx, B, tol=tol, rev_tol=rev_tol,
                                                                            maxiter=maxiter, manifold=manifold)
        n_evals['jacobian'] += n_jac_evals
        if LEAPFROG_SUCCESSFUL:
            logp_p = manifold.log_post(xp)
            n_evals['density'] += 1
            log_ar = logp_p - logp_x - (vp @ vp) / 2 + (v @ v) / 2
            ap = np.exp(np.clip(log_ar, a_min=None, a_max=0.0))
            esjd += ap * np.linalg.norm(xp - x) ** 2
            if logu[i] <= log_ar:
                # Accept
                accepted[i - 1] = 1
                x, logp_x, Jx = xp, logp_p, Jp
                samples[i, :] = xp
            else:
                # Reject
                samples[i, :] = x
                accepted[i - 1] = 0
        else:
            # Reject
            samples[i, :] = x
            accepted[i - 1] = 0
    return samples, n_evals, accepted.mean(), esjd, time.time() - start_time
