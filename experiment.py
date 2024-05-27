import pickle
import numpy as np
from bip import find_point_on_theta_manifold, log_post, grad_forward, find_point_on_xi_manifold
from tangential_hug import thug
from constrained_rwm import crwm
import tensorflow_probability as tfp
import arviz
from manifolds import BIPManifold


def compute_min_ess(samples):
    """Minimum ESS across components."""
    return min(tfp.mcmc.effective_sample_size(samples).numpy())


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
    n_chains = 12
    seed = 1111
    rng = np.random.default_rng(seed=seed)

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
            # Tangential HUG
            S['thug'][six, cix], AP['thug'][six, cix], ESJD['thug'][six, cix], TIME['thug'][six, cix] = thug(
                x0=thetas[cix], T=tau, B=B, N=N, alpha=0.0, logpi=lambda theta: log_post(theta, sigma=sigma),
                grad_f=grad_forward, rng=rng)
            print("\t\tTHUG. Time {} ESJD {} AP {}".format(
                TIME['thug'][six, cix], ESJD['thug'][six, cix], AP['thug'][six, cix]))
            # C-RWM
            xi = find_point_on_xi_manifold(rng=rng, sigma=sigma, theta_fixed=thetas[cix])
            manifold = BIPManifold(sigma=sigmas[six], y_star=1.0)
            S['crwm'][six, cix], _, AP['crwm'][six, cix], ESJD['crwm'][six, cix], TIME['crwm'][six, cix] = crwm(
                x0=xi, manifold=manifold, n=N, T=B*step, B=B, tol=1e-14, rev_tol=1e-14, maxiter=50, rng=rng)
            print("\t\tCRWM. Time {} ESJD {} AP {}".format(
                TIME['crwm'][six, cix], ESJD['crwm'][six, cix], AP['crwm'][six, cix]))

        # Compute ESS with arviz
        for alg in ['thug', 'crwm']:
            ESS_AZ[six] = compute_min_ess_arviz(S[alg][six])
            CC_AZ[six] = ESS_AZ[six] / TIME[alg][six].mean()
            print("\t\t{} ESS-AZ {}".format(alg.upper(), ESS_AZ[six]))
            print("\t\t{} CC-AZ  {}".format(alg.upper(), CC_AZ[six]))

    results = {'ess-tf': ESS_TF, 'ess-az': ESS_AZ, 'esjd': ESJD, 'time': TIME, 'cc-tf': CC_TF, 'cc-az': CC_AZ, 'ap': AP}
    with open("results/experiment_.pkl", "wb") as file:
        pickle.dump(results, file)
