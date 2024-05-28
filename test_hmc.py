"""
I want to understand why HMC seems to have so many numerical issues.
"""
import pickle
import numpy as np
from bip import find_point_on_theta_manifold, log_post, grad_forward, find_point_on_xi_manifold, grad_neg_log_post
from hamiltonian_monte_carlo import hmc
import arviz
from manifolds import BIPManifold


def compute_min_ess_arviz(chains):
    """Expects chains of shape (n_chains, n_samples, dimension)."""
    return np.min(np.array(arviz.ess(arviz.convert_to_dataset(chains)).to_array()) / chains.shape[0])


if __name__ == "__main__":
    # Settings
    N = 2500      # Number of samples
    step = 0.1    # Step size
    B = 20        # Number of integration steps
    tau = B*step  # Total integration time
    sigmas = np.logspace(start=0.0, stop=-6.0, num=7, endpoint=True, base=10.0, dtype=np.float64)  # noise scales
    ns = len(sigmas)
    n_chains = 20
    seed = 1234
    rng = np.random.default_rng(seed=seed)

    # Avoid warnings due to tensorflow's computation of the ESS
    _ = np.seterr(invalid='ignore', over='ignore')

    # Storage
    ESS_TF = {'thug': np.zeros((ns, n_chains)), 'hmc': np.zeros((ns, n_chains)), 'crwm': np.zeros((ns, n_chains))}
    ESS_AZ = {'thug': np.zeros(ns), 'hmc': np.zeros(ns), 'crwm': np.zeros(ns)}
    ESJD = {'thug': np.zeros((ns, n_chains)), 'hmc': np.zeros((ns, n_chains)), 'crwm': np.zeros((ns, n_chains))}
    TIME = {'thug': np.zeros((ns, n_chains)), 'hmc': np.zeros((ns, n_chains)), 'crwm': np.zeros((ns, n_chains))}
    CC_TF = {'thug': np.zeros((ns, n_chains)), 'hmc': np.zeros((ns, n_chains)), 'crwm': np.zeros((ns, n_chains))}
    CC_AZ = {'thug': np.zeros(ns), 'hmc': np.zeros(ns), 'crwm': np.zeros(ns)}
    AP = {'thug': np.zeros((ns, n_chains)), 'hmc': np.zeros((ns, n_chains)), 'crwm': np.zeros((ns, n_chains))}
    S = {
        'thug': np.zeros((ns, n_chains, N, 2)),
        'hmc': np.zeros((ns, n_chains, N, 2)),
        'crwm': np.zeros((ns, n_chains, N, 3))
    }  # samples

    # Find initial points on theta manifold
    thetas = np.vstack([find_point_on_theta_manifold(rng=rng) for _ in range(n_chains)])

    for six, sigma in enumerate(sigmas):
        print("Noise scale: ", sigma)
        for cix in range(n_chains):
            print("\tChain: ", cix)
            # HMC
            S['hmc'][six, cix], AP['hmc'][six, cix], ESJD['hmc'][six, cix], TIME['hmc'][six, cix] = hmc(
                x0=thetas[cix], L=B, step=step, N=N, log_dens=lambda theta: log_post(theta, sigma=sigma),
                gnld=lambda theta: grad_neg_log_post(theta, sigma=sigma), rng=rng)

        # Compute ESS with arviz
        for alg in ['hmc']:
            ESS_AZ[alg][six] = compute_min_ess_arviz(S[alg][six])
            CC_AZ[alg][six] = ESS_AZ[alg][six] / TIME[alg][six].mean()
            print("\t\t{} ESS-AZ {} CC-AZ {}".format(alg.upper(), ESS_AZ[alg][six], CC_AZ[alg][six]))
