import pickle
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from tangential_hug import thug, thug_and_snug, thug_and_rwm
from hamiltonian_monte_carlo import hmc
from bip import find_point_on_theta_manifold, log_post, grad_forward, constraint, grad_neg_log_post
import tensorflow_probability as tfp


if __name__ == "__main__":
    # Settings
    step = 0.1
    B = 20
    N = 2500
    seed = 1234
    sigma = 1.0
    rng = np.random.default_rng(seed=seed)

    # Initial point
    theta = find_point_on_theta_manifold(rng=rng)

    # THUG samples
    s_thug, ap_thug, esjd_thug, runtime_thug = thug(
        x0=theta, step_size=step, B=B, N=N, alpha=0.0,
        log_dens=lambda t: log_post(t, sigma=sigma), grad_f=grad_forward, rng=rng)
    ess_thug = min(tfp.mcmc.effective_sample_size(s_thug))
    print("THUG ESJD: ", esjd_thug)
    # HMC samples
    _ = np.seterr(invalid='ignore', over='ignore')
    s_hmc, ap_hmc, esjd_hmc, runtime_hmc = hmc(
            x0=theta, L=B, step=step, N=N, log_dens=lambda t: log_post(t, sigma=sigma),
            gnld=lambda t: grad_neg_log_post(t, sigma), rng=rng)
    ess_hmc = min(tfp.mcmc.effective_sample_size(s_hmc))
    print("HMC ESJD: ", esjd_hmc)
    # THUG + SNUG, PROB(THUG) = 0.1
    s_ts01, ap_ts01, esjd_ts01, runtime_ts01 = thug_and_snug(
        x0=theta, step_thug=step, step_snug=step, p_thug=0.1, B=B, N=N, alpha=0.0,
        log_dens=lambda t: log_post(t, sigma=sigma), grad_f=grad_forward, rng=rng)
    ess_ts01 = min(tfp.mcmc.effective_sample_size(s_ts01))
    print("THUG+SNUG_01 ESJD: ", esjd_ts01)
    # THUG + SNUG, PROB(THUG) = 0.5
    s_ts05, ap_ts05, esjd_ts05, runtime_ts05 = thug_and_snug(
        x0=theta, step_thug=step, step_snug=step, p_thug=0.5, B=B, N=N, alpha=0.0,
        log_dens=lambda t: log_post(t, sigma=sigma), grad_f=grad_forward, rng=rng)
    ess_ts05 = min(tfp.mcmc.effective_sample_size(s_ts05))
    print("THUG+SNUG_05 ESJD: ", esjd_ts05)
    # THUG + SNUG, PROB(THUG) = 0.9
    s_ts08, ap_ts08, esjd_ts08, runtime_ts08 = thug_and_snug(
        x0=theta, step_thug=step, step_snug=step, p_thug=0.8, B=B, N=N, alpha=0.0,
        log_dens=lambda t: log_post(t, sigma=sigma), grad_f=grad_forward, rng=rng)
    ess_ts08 = min(tfp.mcmc.effective_sample_size(s_ts08))
    print("THUG+SNUG_08 ESJD: ", esjd_ts08)
    # THUG + RWM, PROB(THUG) = 0.5
    s_tr05, ap_tr05, esjd_tr05, runtime_tr05 = thug_and_rwm(
        x0=theta, step_thug=step, step_rwm=0.5*step, p_thug=0.5, B=B, N=N, alpha=0.0,
        log_dens=lambda t: log_post(t, sigma=sigma), grad_f=grad_forward, rng=rng)
    ess_tr05 = min(tfp.mcmc.effective_sample_size(s_tr05))
    print("THUG+RWM ESJD: ", esjd_tr05)

    # Density grid
    xx, yy = np.linspace(-2, 2, 100), np.linspace(-2, 2, 100)
    XX, YY = np.meshgrid(xx, yy)
    pos = np.dstack((XX, YY))
    ZZ = np.apply_along_axis(constraint, 2, pos)

    # Plot samples
    rc('font', **{'family': 'STIXGeneral'})
    fig, ax = plt.subplots(figsize=(20, 4), ncols=5)
    n_std = 15
    ss = 12
    titles = ["THUG", "HMC", "THUG+RWM", "0.5 THUG, 0.5 SNUG", "0.8 THUG, 0.2 SNUG"]   # "0.1 THUG, 0.9 SNUG"
    levels = sigma * np.arange(-n_std, n_std+1)
    for i in range(5):
        contours = ax[i].contourf(XX, YY, ZZ, levels=levels, cmap='viridis')
        if i == 5:
            ax[-1].set_colorbar(contours,
                                label="Standard deviations from " + r"$\mathregular{C(\theta) = F(\theta) - y = 0}$")
        ax[i].contour(XX, YY, ZZ, levels=[0.0], colors='black')
    ax[0].scatter(*s_thug.T, s=ss, color='dodgerblue', ec='navy', label=f"{ess_thug:.2f}")
    ax[1].scatter(*s_hmc.T, s=ss, color='lightcoral', ec='brown', label=f"{ess_hmc:.2f}")
    # ax[2].scatter(*s_ts01.T, s=ss, color='lawngreen', ec='olivedrab', label=f"{ess_ts01:.2f}")
    ax[2].scatter(*s_tr05.T, s=ss, color='lawngreen', ec='olivedrab', label=f"{ess_tr05:.2f}")
    ax[3].scatter(*s_ts05.T, s=ss, color='thistle', ec='purple', label=f"{ess_ts05:.2f}")
    ax[4].scatter(*s_ts08.T, s=ss, color='gold', ec='goldenrod', label=f"{ess_ts08:.2f}")
    for i in range(5):
        ax[i].set_aspect('equal')
        ax[i].set_xlabel(r"$\mathregular{\theta_0}$", fontsize=15)
        ax[i].set_title(titles[i])
        ax[i].legend()
    ax[0].set_ylabel(r"$\mathregular{\theta_1}$", fontsize=15)
    plt.tight_layout()
    plt.legend(fontsize=12)
    # plt.savefig("images/various_algorithms_samples_sigma_{}.png".format(
    #     str(sigma).replace('.', 'dot')), dpi=300)
    plt.show()

