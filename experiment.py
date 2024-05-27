import pickle
import numpy as np
from bip import find_point_on_theta_manifold, log_post, grad_forward, find_point_on_xi_manifold, grad_neg_log_post
from tangential_hug import thug
from constrained_rwm import crwm
from hamiltonian_monte_carlo import hmc
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
            # Tangential HUG
            # if six > 0:
            S['thug'][six, cix], AP['thug'][six, cix], ESJD['thug'][six, cix], TIME['thug'][six, cix] = thug(
                x0=thetas[cix], T=tau, B=B, N=N, alpha=0.0, logpi=lambda theta: log_post(theta, sigma=sigma),
                grad_f=grad_forward, rng=rng)
            # else:
            # S['thug'][six, cix], AP['thug'][six, cix], ESJD['thug'][six, cix], TIME['thug'][six, cix] = ts(
            #     x0=thetas[cix], T=B, thug_step=step, snug_step=step, N=N, alpha=0.0,
            #     logpi=lambda theta: log_post(theta, sigma=sigma), grad_f=grad_forward, rng=rng, p_thug=0.8)
            # ESS_TF['thug'][six, cix] = compute_min_ess(S['thug'][six, cix])
            # CC_TF['thug'][six, cix] = ESS_TF['thug'][six, cix] / TIME['thug'][six, cix]
            # print("\t\tTHUG. Time {} ESS {} CC {} ESJD {} AP {}".format(
            #     TIME['thug'][six, cix], ESS_TF['thug'][six, cix], CC_TF['thug'][six, cix],
            #     ESJD['thug'][six, cix], AP['thug'][six, cix]))
            # # C-RWM
            xi = find_point_on_xi_manifold(rng=rng, sigma=sigma, theta_fixed=thetas[cix])
            manifold = BIPManifold(sigma=sigmas[six], y_star=1.0)
            S['crwm'][six, cix], _, AP['crwm'][six, cix], ESJD['crwm'][six, cix], TIME['crwm'][six, cix] = crwm(
                x0=xi, manifold=manifold, n=N, T=B*step, B=B, tol=1e-14, rev_tol=1e-14, maxiter=50, rng=rng)
            # print("\t\tCRWM. Time {} ESJD {} AP {}".format(
            #     TIME['crwm'][six, cix], ESJD['crwm'][six, cix], AP['crwm'][six, cix]))
            # # HMC
            S['hmc'][six, cix], AP['hmc'][six, cix], ESJD['hmc'][six, cix], TIME['hmc'][six, cix] = hmc(
                x0=thetas[cix], L=B, step=step, N=N, log_dens=lambda theta: log_post(theta, sigma=sigma),
                gnld=lambda theta: grad_neg_log_post(theta, sigma), rng=rng)
            # ESS_TF['hmc'][six, cix] = compute_min_ess(S['hmc'][six, cix])
            # CC_TF['hmc'][six, cix] = ESS_TF['hmc'][six, cix] / TIME['hmc'][six, cix]
            # print("\t\tHMC. Time {} ESS {} CC {} ESJD {} AP {}".format(
            #     TIME['hmc'][six, cix], ESS_TF['hmc'][six, cix], CC_TF['hmc'][six, cix],
            #     ESJD['hmc'][six, cix], AP['hmc'][six, cix]))

        # Compute ESS with arviz
        for alg in ['thug', 'crwm', 'hmc']:
            ESS_AZ[alg][six] = compute_min_ess_arviz(S[alg][six])
            CC_AZ[alg][six] = ESS_AZ[alg][six] / TIME[alg][six].mean()
            print("\t\t{} ESS-AZ {} CC-AZ {}".format(alg.upper(), ESS_AZ[alg][six], CC_AZ[alg][six]))

    results = {'ess-tf': ESS_TF, 'ess-az': ESS_AZ, 'esjd': ESJD, 'time': TIME, 'cc-tf': CC_TF, 'cc-az': CC_AZ, 'ap': AP}
    with open("results/experiment{}.pkl".format(seed), "wb") as file:
        pickle.dump(results, file)
