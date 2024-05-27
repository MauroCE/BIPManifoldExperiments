import pickle
import time
import arviz
from concurrent.futures import ThreadPoolExecutor
import numpy
import scipy as sp
from warnings import catch_warnings, filterwarnings
import torch


device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")
print("Device: ", device)


# ----------------- UTILITY -------------------
def compute_min_ess_arviz(chains):
    """Expects chains of shape (n_chains, n_samples, dimension)."""
    return numpy.min(numpy.array(arviz.ess(arviz.convert_to_dataset(chains.numpy())).to_array()) / chains.shape[0])


# ----------------- BIP FUNCTIONS -------------------
def forward(theta):
    """Forward function for the inverse problem, mapping from R^2 to R."""
    return theta[1]**2 + 3.0*(theta[0]**2)*(theta[0]**2 - 1.0)


def constraint(theta, y=1.0):
    """Constraint function. For these experiments we always consider observed data to be y=1."""
    return forward(theta) - y


def grad_forward(theta, dev=None):
    """Gradient of the forward function."""
    return torch.tensor([12.0*theta[0]**3 - 6.0*theta[0], 2.0*theta[1]], device=dev)


def log_prior(theta):
    """Log prior for theta is a standard normal."""
    return -0.5*torch.sum(theta**2)


def log_post(theta, noise_scale):
    """Log posterior for theta given a noise scale sigma."""
    return log_prior(theta) - 0.5*(constraint(theta)**2)/(noise_scale**2)


def grad_neg_log_post(theta, noise_scale, dev=None):
    """Gradient of the negative log posterior for theta."""
    return theta + constraint(theta)*grad_forward(theta, dev=dev)/noise_scale**2


def find_point_on_theta_manifold(maxiter=1000, tol=1e-12, y=1.0, generator=None):
    """Finds a point on the theta manifold."""
    iteration = 0
    with catch_warnings():
        filterwarnings('error')
        while iteration <= maxiter:
            iteration += 1
            try:
                theta0 = torch.randn(1, generator=generator).numpy()
                theta1_sol = sp.optimize.fsolve(
                    func=lambda theta1: constraint(torch.tensor(numpy.concatenate((theta0, theta1))), y=y).numpy(),
                    x0=torch.randn(1, generator=g).numpy()
                )
                theta_found = torch.tensor([theta0[0], *theta1_sol])
                if abs(constraint(theta_found, y=y)) <= tol:
                    return theta_found
            except RuntimeWarning:
                continue


# ----------------- TANGENTIAL HUG -------------------
def thug(x0, step_size, n_int, n_samples, alpha, logpi, grad_f, generator, dev):
    """
    Tangential Hug. Notice that it doesn't matter whether we use the gradient of pi or
    grad log pi to tilt the velocity.
    """
    start_time = time.time()
    # Grab dimension, initialize storage for samples & acceptances
    samples = torch.zeros((N, len(x0)), device=dev)
    acceptances = torch.zeros(N, device=dev)
    esjd = 0.0  # rao-blackwellised
    for i in range(n_samples):
        v0s = torch.randn(x0.shape, generator=generator, device=dev)  # Draw velocity spherically
        g = grad_f(x0, device=dev)              # Compute gradient at x0
        g = g / torch.norm(g)                  # Normalize
        v0 = v0s - alpha * g * (g @ v0s)  # Tilt velocity
        v, x = v0, x0                    # Housekeeping
        logu = torch.log(torch.rand(1, generator=generator, device=dev))  # Acceptance ratio

        for _ in range(n_int):
            x = x + step_size*v/2           # Move to midpoint
            g = grad_f(x)          # Compute gradient at midpoint
            ghat = g / torch.norm(g)          # Normalize
            v = v - 2*(v @ ghat) * ghat  # Reflect velocity using midpoint gradient
            x = x + step_size*v/2           # Move from midpoint to end-point
        # Unsqueeze the velocity
        g = grad_f(x, device=dev)
        g = g / torch.norm(g)
        v = v + (alpha / (1 - alpha)) * g * (g @ v)
        # In the acceptance ratio must use spherical velocities, hence v0s and the unsqueezed v
        log_ar = logpi(x) - 0.5*(v@v) - logpi(x0) + 0.5*(v0s@v0s)
        ap = torch.exp(torch.clip(log_ar, min=None, max=0.0))  # acceptance probability
        esjd += ap*torch.norm(x0 - x)**2
        if logu <= log_ar:
            acceptances[i] = 1         # Accepted!
            x0 = x
        samples[i] = x0
    return samples, acceptances.mean(), esjd, time.time() - start_time


def run_thug(s_ix, c_ix, noise_scale, theta_list, step_size, n_int, n_samples, generator, dev):
    """Runs THUG."""
    s, ap, esjd, runtime = thug(
        x0=theta_list[c_ix], step_size=step_size, n_int=n_int, n_samples=n_samples, alpha=0.0,
        logpi=lambda theta: log_post(theta, noise_scale=noise_scale), grad_f=grad_forward, generator=generator, dev=dev)
    return s_ix, c_ix, s, ap, esjd, runtime


# ----------------- HAMILTONIAN MONTE CARLO -------------------

def leapfrog(x, v, gnld, step_size, n_leapfrogs, dev=None):
    """Leapfrog integrator, assuming an identity mass matrix."""
    assert isinstance(step_size, float), "Step size must be a float."
    assert isinstance(n_leapfrogs, int), "Number of leapfrog steps must be an integer."
    assert n_leapfrogs > 0, "Number of leapfrog steps must be positive."
    assert x.shape == v.shape, "Position and velocity must have the same shape."

    # first half-step for momentum
    v = v - 0.5*step*gnld(x, dev=dev)

    # n_leapfrog - 1 full steps for both position and velocity
    for _ in range(n_leapfrogs - 1):
        x = x + step*v
        v = v - step*gnld(x, dev=dev)

    # Final full position step and half-momentum step
    x = x + step*v
    v = v - 0.5*step*gnld(x, dev=dev)

    return x, -v  # reverse velocity for completeness


def hmc(x0, n_leapfrog, step_size, n_samples, log_dens, gnld, dev=None, generator=None):
    """Hamiltonian Monte Carlo with identity mass matrix."""
    start_time = time.time()
    samples = torch.zeros((n_samples, len(x0)), device=dev)
    acceptances = torch.zeros(n_samples)
    esjd = 0.0  # rao-blackwellised
    for i in range(n_samples):
        # Sample a new velocity
        v0 = torch.randn(len(x0), generator=generator, device=dev)
        # Leapfrog
        xL, vL = leapfrog(x=x0, v=v0, gnld=gnld, step_size=step_size, n_leapfrogs=n_leapfrog)
        # Compute acceptance probability
        log_ar = log_dens(xL) - log_dens(x0) - 0.5*(vL@vL) + 0.5*(v0@v0)
        ap = torch.exp(torch.clip(log_ar, min=None, max=0.0))
        esjd += ap*torch.norm(x0 - xL)**2
        if torch.log(torch.rand(1, generator=generator, device=dev)) <= log_ar:
            acceptances[i] = 1
            x0 = xL
        samples[i] = x0
    return samples, acceptances.mean(), esjd, time.time() - start_time


def run_hmc(s_ix, c_ix, noise_scale, theta_list, step_size, n_int, n_samples, generator, dev):
    """Run HMC."""
    s, ap, esjd, runtime = hmc(
        x0=theta_list[c_ix], n_leapfrog=n_int, step_size=step_size, n_samples=n_samples,
        log_dens=lambda theta: log_post(theta, noise_scale=noise_scale),
        gnld=lambda theta: grad_neg_log_post(theta, noise_scale=noise_scale), dev=dev, generator=generator)
    return s_ix, c_ix, s, ap, esjd, runtime


def run_hmc_prop(s_ix, c_ix, noise_scale, theta_list, int_time, n_int, n_samples, generator, dev):
    """Run HMC with proportional step size."""
    B_new = (10**s_ix) * n_int
    step_new = int_time / B_new
    s, ap, esjd, runtime = hmc(
        x0=theta_list[c_ix], n_leapfrog=B_new, step_size=step_new, n_samples=n_samples,
        log_dens=lambda theta: log_post(theta, noise_scale=noise_scale),
        gnld=lambda theta: grad_neg_log_post(theta, noise_scale=noise_scale), dev=dev, generator=generator)
    return s_ix, c_ix, s, ap, esjd, time, B_new, step_new


# ----------------- CONSTRAINED RANDOM WALK -------------------
def qr_project(v, J, Q_out, R_out):
    """Projects onto the tangent space using QR decomposition."""
    torch.linalg.qr(J.T, mode='reduced', out=(Q_out, R_out))
    return Q_out @ (Q_out.T @ v)


def constrained_rwm_step(x, v, tol, maxiter, Jx, noise_scale, dev=None):
    """Used for both forward and backward. See Manifold-Lifting paper."""
    # Project momentum
    Q_out = torch.empty(Jx.shape[1], Jx.shape[0], device=dev)
    R_out = torch.empty(Jx.shape[0], Jx.shape[0], device=dev)
    v_projected = v - qr_project(v, Jx, Q_out, R_out)
    # Unconstrained position step
    x_unconstr = x + v_projected
    # Position Projection
    a, flag = project_zappa_manifold(x_unconstr, Jx.T, tol, maxiter)
    y = x_unconstr - Jx.T @ a
    try:
        Jy = jacobian(y, noise_scale, dev=dev)
    except ValueError as e:
        print("Jacobian computation at projected point failed. ", e)
        return x, v, Jx, 0
    # backward velocity
    v_would_have = y - x
    # Find backward momentum & project it to tangent space at new position
    Q_out = torch.empty(Jy.shape[1], Jy.shape[0], device=dev)
    R_out = torch.empty(Jy.shape[0], Jy.shape[0], device=dev)
    v_projected_endposition = v_would_have - qr_project(v_would_have, Jy, Q_out, R_out)
    # Return projected position, projected momentum and flag
    return y, v_projected_endposition, Jy, flag


def constrained_leapfrog(x0, v0, J0, n_int, tol, rev_tol, maxiter, noise_scale, dev=None):
    """Constrained Leapfrog/RATTLE."""
    successful = True
    x, v, J = x0, v0, J0
    for _ in range(n_int):
        xf, vf, Jf, converged_fw, n_fw = constrained_rwm_step(x, v, tol, maxiter, J, noise_scale, dev=dev)
        xr, vr, Jr, converged_bw, n_bw = constrained_rwm_step(xf, -vf, tol, maxiter, Jf, noise_scale, dev=dev)
        if (not converged_fw) or (not converged_bw) or (torch.norm(xr - x) >= rev_tol):
            successful = False
            return x0, v0, J0, successful
        else:
            x = xf
            v = vf
            J = Jf
    return x, v, J, successful


def constraint_ext(xi, y_star):
    """Constraint function for the extended space."""
    return [xi[1]**2 + 3*xi[0]**2*(xi[0]**2 - 1)] + xi[2] - y_star


def project_zappa_manifold(z, Q, y_star, noise_scale, dev=None, tol=1.48e-08, maxiter=50):
    """
    This version is the version of Miranda & Zappa. It returns i, the number of iterations
    i.e. the number of gradient evaluations used.
    """
    a, flag, i = torch.zeros(Q.shape[1], device=dev), 1, 0

    # Compute the constrained at z - Q@a. If it fails due to overflow error, return a rejection altogether.
    try:
        projected_value = constraint_ext(z - Q @ a, y_star)
    except ValueError:
        return a, 0
    # While loop
    while torch.norm(projected_value) >= tol:
        try:
            Jproj = jacobian(z - Q @ a, noise_scale, dev=dev)
        except ValueError as e:
            print("Jproj failed. ", e)
            return torch.zeros(Q.shape[1], device=dev), 0
        # Check that Jproj@Q is invertible. Do this by checking condition number
        # see https://stackoverflow.com/questions/13249108/efficient-pythonic-check-for-singular-matrix
        GramMatrix = Jproj @ Q
        if torch.linalg.cond(GramMatrix) < 1 / torch.finfo(z.dtype).eps:
            delta_a = torch.empty(Q.shape[1], device=dev)
            delta_a = torch.linalg.solve(GramMatrix, projected_value, out=delta_a)
            a += delta_a
            i += 1
            if i > maxiter:
                return torch.zeros(Q.shape[1], device=dev), 0
            # If we are not at maxiter iteration, compute new projected value
            try:
                projected_value = constraint_ext(z - Q @ a, y_star)
            except ValueError:
                return torch.zeros(Q.shape[1], device=dev), 0
        else:
            # Fail
            return torch.zeros(Q.shape[1], device=dev), 0
    return a, 1


def jacobian(xi, noise_scale, dev=None):
    return torch.tensor([12 * xi[0] ** 3 - 6 * xi[0], 2 * xi[1], noise_scale], device=dev).reshape(1, -1)


def log_post_ext(xi, noise_scale, dev=None):
    """log posterior for c-rwm"""
    jac = jacobian(xi, noise_scale, dev=dev)
    return - xi[:2]@xi[:2]/2 - xi[-1]**2/2 - torch.log(jac@jac.T + noise_scale**2)/2


def crwm(x0, n, int_time, n_int, tol, rev_tol, noise_scale, maxiter=50, generator=None, dev=None):
    """C-RWM using RATTLE."""
    start_time = time.time()
    assert isinstance(n_int, int), "Number of integration steps B must be an integer."
    assert isinstance(n, int), "Number of samples n must be an integer."
    # Settings
    step_size = int_time / n_int
    d = 2
    m = 1

    # Initial point on the manifold
    x = x0

    # Storage
    samples = torch.zeros((n, d + m), device=dev)  # Store n samples on the manifold
    samples[0, :] = x
    accepted = torch.zeros(n, device=dev)
    esjd = 0.0                          # rao-blackwellised expected square jump distance

    # Log-uniforms for MH accept-reject step
    logu = torch.log(torch.randn(n, generator=generator, device=dev))

    # Compute jacobian & density value
    Jx = jacobian(x, noise_scale=sigma, dev=dev)
    logp_x = log_post_ext(x, noise_scale=noise_scale, dev=dev)

    for i in range(n):
        v = step_size * torch.randn(m+d, generator=generator, device=dev)  # Sample in the ambient space.
        xp, vp, Jp, LEAPFROG_SUCCESSFUL = constrained_leapfrog(
            x, v, Jx, B, tol=tol, rev_tol=rev_tol, maxiter=maxiter, noise_scale=noise_scale, dev=dev)
        if LEAPFROG_SUCCESSFUL:
            logp_p = log_post_ext(xp, noise_scale, dev=dev)
            log_ar = logp_p - logp_x - (vp @ vp) / 2 + (v @ v) / 2
            ap = torch.exp(torch.clip(log_ar, min=None, max=0.0))
            esjd += ap * torch.norm(xp - x) ** 2
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
    return samples, accepted.mean(), esjd, time.time() - start_time


def run_crwm(s_ix, c_ix, noise_scale, theta_list, step_size, n_int, n_samples, generator, dev):
    s, ap, esjd, runtime = crwm(
        x0=theta_list[c_ix], n=n_samples, int_time=step_size*n_int, n_int=n_int, tol=1e-14, rev_tol=1e-14,
        noise_scale=noise_scale, generator=generator, dev=dev)
    return s_ix, c_ix, s, ap, esjd, runtime


if __name__ == "__main__":
    # Settings
    seed = 1234
    g = torch.Generator(device=device).manual_seed(seed)
    sigmas = 10**torch.linspace(start=0.0, end=-5.0, steps=6, device=device)
    ns = len(sigmas)
    n_chains = 12
    N = 2500
    B = 20
    step = 0.1
    tau = B*step
    # Storage
    ESS_AZ = {
        'thug': torch.zeros(ns, device=device),
        'hmc': torch.zeros(ns, device=device),
        'crwm': torch.zeros(ns, device=device)}
    ESJD = {
        'thug': torch.zeros((ns, n_chains), device=device),
        'hmc': torch.zeros((ns, n_chains), device=device),
        'crwm': torch.zeros((ns, n_chains), device=device)}
    TIME = {
        'thug': torch.zeros((ns, n_chains), device=device),
        'hmc': torch.zeros((ns, n_chains), device=device),
        'crwm': torch.zeros((ns, n_chains), device=device)}
    CC_AZ = {
        'thug': torch.zeros(ns, device=device),
        'hmc': torch.zeros(ns, device=device),
        'crwm': torch.zeros(ns, device=device)}
    AP = {
        'thug': torch.zeros((ns, n_chains), device=device),
        'hmc': torch.zeros((ns, n_chains), device=device),
        'crwm': torch.zeros((ns, n_chains), device=device)}
    S = {
        'thug': torch.zeros((ns, n_chains, N, 2), device=device),
        'hmc': torch.zeros((ns, n_chains, N, 2), device=device),
        'crwm': torch.zeros((ns, n_chains, N, 2), device=device)}  # samples
    # Initial points on the manifold
    thetas = torch.stack([find_point_on_theta_manifold(generator=g) for _ in range(n_chains)])

    with ThreadPoolExecutor() as executor:
        futures_thug = []
        futures_hmc = []
        futures_crwm = []
        futures_hmc_prop = []
        for six, sigma in enumerate(sigmas):
            for cix in range(n_chains):
                # THUG
                futures_thug.append(executor.submit(run_thug, six, cix, sigma, thetas, step, B, N, g, dev=device))
                # HMC
                futures_hmc.append(executor.submit(run_hmc, six, cix, sigma, thetas, tau, B, N, g, dev=device))
                # C-RWM
                futures_crwm.append(executor.submit(run_crwm, six, cix, sigma, thetas, step, B, N, g, dev=device))
        # Unpack results for THUG
        for future in futures_thug:
            six, cix, s, ap, esjd, runtime = future.result()
            label = 'thug'
            S[label][six, cix] = s
            AP[label][six, cix] = ap
            ESJD[label][six, cix] = esjd
            TIME[label][six, cix] = runtime
            print("\t\tTHUG finished in {:.5f}s.".format(runtime))
        # Unpack results for HMC
        for future in futures_hmc:
            six, cix, s, ap, esjd, runtime = future.result()
            label = 'hmc'
            S[label][six, cix] = s
            AP[label][six, cix] = ap
            ESJD[label][six, cix] = esjd
            TIME[label][six, cix] = runtime
            print("\t\tTHUG finished in {:.5f}s.".format(runtime))
        # Unpack results for C-RWM
        for future in futures_crwm:
            six, cix, s, ap, esjd, runtime = future.result()
            label = 'crwm'
            S[label][six, cix] = s
            AP[label][six, cix] = ap
            ESJD[label][six, cix] = esjd
            TIME[label][six, cix] = runtime
            print("\t\tCRWM finished in {:.5f}s.".format(runtime))

    # Compute ESS with arviz
    for alg in ['thug', 'crwm', 'hmc']:
        ESS_AZ[alg][six] = torch.tensor(compute_min_ess_arviz(S[alg][six]))
        CC_AZ[alg][six] = ESS_AZ[alg][six] / TIME[alg][six].mean()
        print("\t\t{} ESS-AZ {} CC-AZ {}".format(alg.upper(), ESS_AZ[alg][six], CC_AZ[alg][six]))

    # Save data
    results = {'ess-az': ESS_AZ, 'esjd': ESJD, 'time': TIME, 'cc-az': CC_AZ, 'ap': AP}
    with open("results/experiment{}_gpu.pkl".format(seed), "wb") as file:
        pickle.dump(results, file)

