"""
Here I will use the CPU and parallelize all computations to speed things up and make sure that all algorithms have
the same initial cost for the parallel set up. Runs THUG, HMC, CRWM and HMC-PROP in parallel.
"""
import pickle
import numpy as np
from bip import find_point_on_theta_manifold, log_post, grad_neg_log_post, grad_forward, jac_forward_extended
from bip import log_post_extended, constraint_extended, find_point_on_xi_manifold
from hamiltonian_monte_carlo import hmc
from constrained_rwm import crwm
from tangential_hug import thug
import arviz
from concurrent.futures import ProcessPoolExecutor
_ = np.seterr(invalid="ignore", over="ignore")


def compute_min_ess_arviz(chains):
    """Expects chains of shape (n_chains, n_samples, dimension)."""
    return np.min(np.array(arviz.ess(arviz.convert_to_dataset(chains)).to_array()) / chains.shape[0])


def run_hmc(s_ix, c_ix, noise_scale, theta_list, int_time, n_int, n_samples, generator, prop_step=False):
    """Run HMC."""
    if prop_step:
        n_int = (10**s_ix) * n_int
    step_size = int_time / n_int
    _samples, _ap, _esjd, _runtime = hmc(
        x0=theta_list[c_ix], L=n_int, step=step_size, N=n_samples,
        log_dens=lambda theta: log_post(theta, sigma=noise_scale),
        gnld=lambda theta: grad_neg_log_post(theta, noise_scale), rng=generator)
    return s_ix, c_ix, _samples, _ap, _esjd, _runtime


def run_thug(s_ix, c_ix, noise_scale, theta_list, step_size, n_int, n_samples, generator):
    """Run HMC."""
    _samples, _ap, _esjd, _runtime = thug(
        x0=theta_list[c_ix], step_size=step_size, B=n_int, N=n_samples, alpha=0.0,
        log_dens=lambda theta: log_post(theta, sigma=noise_scale), grad_f=grad_forward, rng=generator)
    return s_ix, c_ix, _samples, _ap, _esjd, _runtime


def run_crwm(s_ix, c_ix, noise_scale, xi_list, step_size, n_int, n_samples, generator):
    _samples, _evals, _ap, _esjd, _runtime = crwm(
        x0=xi_list[c_ix, s_ix], log_dens=lambda xi: log_post_extended(xi, noise_scale),
        jac=lambda xi: jac_forward_extended(xi, noise_scale),
        constraint=lambda xi: constraint_extended(xi, noise_scale), n=n_samples, T=step_size*n_int, B=n_int, tol=1e-14,
        rev_tol=1e-14, rng=generator)
    return s_ix, c_ix, _samples, _ap, _esjd, _runtime


if __name__ == "__main__":
    # Settings
    N = 2500      # Number of samples
    step = 0.5    # Step size
    B = 10        # Number of integration steps
    tau = B*step  # Total integration time
    sigmas = np.logspace(start=0.0, stop=-5.0, num=6, endpoint=True, base=10.0, dtype=np.float64)  # noise scales
    ns = len(sigmas)
    n_chains = 11
    seed = 1234
    rng = np.random.default_rng(seed=seed)

    # Avoid warnings due to tensorflow's computation of the ESS
    _ = np.seterr(invalid='ignore', over='ignore')

    # Storage
    ESS_AZ = {'thug': np.zeros(ns), 'hmc': np.zeros(ns), 'crwm': np.zeros(ns), 'hmc-p': np.zeros(ns)}
    ESJD = {
        'thug': np.zeros((ns, n_chains)),
        'hmc': np.zeros((ns, n_chains)),
        'crwm': np.zeros((ns, n_chains)),
        'hmc-p': np.zeros((ns, n_chains))}
    TIME = {
        'thug': np.zeros((ns, n_chains)),
        'hmc': np.zeros((ns, n_chains)),
        'crwm': np.zeros((ns, n_chains)),
        'hmc-p': np.zeros((ns, n_chains))}
    CC_AZ = {'thug': np.zeros(ns), 'hmc': np.zeros(ns), 'crwm': np.zeros(ns), 'hmc-p': np.zeros(ns)}
    AP = {
        'thug': np.zeros((ns, n_chains)),
        'hmc': np.zeros((ns, n_chains)),
        'crwm': np.zeros((ns, n_chains)),
        'hmc-p': np.zeros((ns, n_chains))}
    S = {  # samples
        'thug': np.zeros((ns, n_chains, N, 2)),
        'hmc': np.zeros((ns, n_chains, N, 2)),
        'crwm': np.zeros((ns, n_chains, N, 3)),
        'hmc-p': np.zeros((ns, n_chains, N, 2))
    }

    # Find initial points on theta manifold
    thetas = np.vstack([find_point_on_theta_manifold(rng=rng) for _ in range(n_chains)])
    # Find corresponding points on xi manifold
    xis = np.full((n_chains, ns, 3), np.nan)
    for cix in range(n_chains):
        for six, sigma in enumerate(sigmas):
            xis[cix, six] = find_point_on_xi_manifold(rng=rng, sigma=sigma, theta_fixed=thetas[cix])

    with (ProcessPoolExecutor() as executor):
        # Run HMC, THUG and CRWM in parallel
        futures_thug = []
        futures_hmc = []
        futures_crwm = []
        for six, sigma in enumerate(sigmas):
            for cix in range(n_chains):
                futures_thug.append(executor.submit(run_thug, six, cix, sigma, thetas, step, B, N, rng))
                futures_hmc.append(executor.submit(run_hmc, six, cix, sigma, thetas, tau, B, N, rng, prop_step=False))
                futures_crwm.append(executor.submit(run_crwm, six, cix, sigma, xis, step, B, N, rng))
        # Get their results
        for alg, future_list in zip(['thug', 'hmc', 'crwm'], [futures_thug, futures_hmc, futures_crwm]):
            for future in future_list:
                out = future.result()
                __six, __cix, __samples, __ap, __esjd, __runtime = out
                S[alg][__six, __cix] = __samples
                AP[alg][__six, __cix] = __ap
                ESJD[alg][__six, __cix] = __esjd
                TIME[alg][__six, __cix] = __runtime
                print("\t\t{} ({}, {}) finished in {:.5f}s.".format(alg.upper(), __six, __cix, TIME[alg][__six, __cix]))
        # Run HMC-PROP in parallel
        futures_hmc_p = []
        for six, sigma in enumerate(sigmas[:4]):
            for cix in range(n_chains):
                futures_hmc_p.append(executor.submit(run_hmc, six, cix, sigma, thetas, tau, B, N, rng, prop_step=True))
        # Get results of HMC-PROP
        for future in futures_hmc_p:
            __six, __cix, __samples, __ap, __esjd, __runtime = future.result()
            S['hmc-p'][__six, __cix] = __samples
            AP['hmc-p'][__six, __cix] = __ap
            ESJD['hmc-p'][__six, __cix] = __esjd
            TIME['hmc-p'][__six, __cix] = __runtime
            print("\t\tHMC-PROP ({}, {}) finished in {:.5f}s.".format(__six, __cix, TIME['hmc-p'][__six, __cix]))

    # Compute ESS with arviz
    for alg in ['thug', 'hmc', 'crwm', 'hmc-p']:
        sigma_list = sigmas if alg != 'hmc-p' else sigmas[:4]
        for six in range(len(sigma_list)):
            ESS_AZ[alg][six] = compute_min_ess_arviz(S[alg][six])
            CC_AZ[alg][six] = ESS_AZ[alg][six] / TIME[alg][six].mean()
            print("\t\t{} ESS-AZ {} CC-AZ {}".format(alg.upper(), ESS_AZ[alg][six], CC_AZ[alg][six]))

    results = {'ess-az': ESS_AZ, 'esjd': ESJD, 'time': TIME, 'cc-az': CC_AZ, 'ap': AP}
    with open("results/experiment{}_final_T{}_step{}.pkl".format(seed, B, str(step).replace('.', '')), "wb") as file:
        pickle.dump(results, file)
