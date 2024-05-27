import pickle
import numpy as np
from bip import find_point_on_theta_manifold, log_post, grad_neg_log_post
from hamiltonian_monte_carlo import hmc
import tensorflow_probability as tfp
import arviz
from concurrent.futures import ProcessPoolExecutor


def compute_min_ess(samples):
    """Minimum ESS across components."""
    return min(tfp.mcmc.effective_sample_size(samples).numpy())


def compute_min_ess_arviz(chains):
    """Expects chains of shape (n_chains, n_samples, dimension)."""
    return np.min(np.array(arviz.ess(arviz.convert_to_dataset(chains)).to_array()) / chains.shape[0])


def run_hmc(six, sigma, cix, thetas, tau, B, N, rng):
    """Run HMC."""
    B_new = (10**six) * B
    step_new = tau / B_new
    S, AP, ESJD, TIME = hmc(
        x0=thetas[cix], L=B_new, step=step_new, N=N, log_dens=lambda theta: log_post(theta, sigma=sigma),
        gnld=lambda theta: grad_neg_log_post(theta, sigma), rng=rng)
    return six, cix, S, AP, ESJD, TIME, B_new, step_new


if __name__ == "__main__":
    # Settings
    N = 2500      # Number of samples
    step = 0.1    # Step size
    B = 20        # Number of integration steps
    tau = B*step  # Total integration time
    sigmas = np.logspace(start=0.0, stop=-5.0, num=6, endpoint=True, base=10.0, dtype=np.float64)  # noise scales
    ns = len(sigmas)
    n_chains = 11
    seed = 1234
    rng = np.random.default_rng(seed=seed)

    # Avoid warnings due to tensorflow's computation of the ESS
    _ = np.seterr(invalid='ignore', over='ignore')

    # Storage
    ESS_TF = np.zeros((ns, n_chains))
    ESS_AZ = np.zeros(ns)
    ESJD = np.zeros((ns, n_chains))
    TIME = np.zeros((ns, n_chains))
    CC_TF = np.zeros((ns, n_chains))
    CC_AZ = np.zeros(ns)
    AP = np.zeros((ns, n_chains))
    S = np.zeros((ns, n_chains, N, 2))  # samples

    # Find initial points on theta manifold
    thetas = np.vstack([find_point_on_theta_manifold(rng=rng) for _ in range(n_chains)])

    with ProcessPoolExecutor() as executor:
        futures = []
        for six, sigma in enumerate(sigmas):
            for cix in range(n_chains):
                futures.append(executor.submit(run_hmc, six, sigma, cix, thetas, tau, B, N, rng))

        for future in futures:
            six, cix, S[six, cix], AP[six, cix], ESJD[six, cix], TIME[six, cix], B_new, step_new = future.result()
            print("\t\tHMC finished in {:.5f}s. Tau: ".format(TIME[six, cix]), B_new * step_new, " Step: ", step_new, " B: ", B_new)

    # Compute ESS with arviz
    for six in range(len(sigmas)):
        ESS_AZ[six] = compute_min_ess_arviz(S[six])
        CC_AZ[six] = ESS_AZ[six] / TIME[six].mean()
        print("\t\t{} ESS-AZ {} CC-AZ {}".format('HMC-PROP', ESS_AZ[six], CC_AZ[six]))

    results = {'ess-tf': ESS_TF, 'ess-az': ESS_AZ, 'esjd': ESJD, 'time': TIME, 'cc-tf': CC_TF, 'cc-az': CC_AZ, 'ap': AP}
    with open("results/experiment{}_hmc_prop_parallel_full.pkl".format(seed), "wb") as file:
        pickle.dump(results, file)
