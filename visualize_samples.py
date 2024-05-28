import pickle
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from tangential_hug import thug
from hamiltonian_monte_carlo import hmc
from bip import find_point_on_theta_manifold, log_post, grad_forward, constraint, grad_neg_log_post


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
    # HMC samples
    _ = np.seterr(invalid='ignore', over='ignore')
    s_hmc, ap_hmc, esjd_hmc, runtime_hmc = hmc(
            x0=theta, L=B, step=step, N=N, log_dens=lambda t: log_post(t, sigma=sigma),
            gnld=lambda t: grad_neg_log_post(t, sigma), rng=rng)

    # Density grid
    xx, yy = np.linspace(-2, 2, 100), np.linspace(-2, 2, 100)
    XX, YY = np.meshgrid(xx, yy)
    pos = np.dstack((XX, YY))
    ZZ = np.apply_along_axis(constraint, 2, pos)

    # Plot samples
    rc('font', **{'family': 'STIXGeneral'})
    fig, ax = plt.subplots(figsize=(5, 5))
    n_std = 15
    levels = sigma * np.arange(-n_std, n_std+1)
    contours = ax.contourf(XX, YY, ZZ, levels=levels, cmap='viridis')
    ax.contour(XX, YY, ZZ, levels=[0.0], colors='black')
    ax.scatter(*s_thug.T, s=4, color='dodgerblue', ec='navy')
    ax.scatter(*s_hmc.T, s=4, color='lightcoral', ec='brown')
    ax.set_aspect('equal')
    ax.set_xlabel(r"$\mathregular{\theta_0}$", fontsize=15)
    ax.set_ylabel(r"$\mathregular{\theta_1}$", fontsize=15)
    plt.colorbar(contours, label="Standard deviations from " + r"$\mathregular{C(\theta) = F(\theta) - y = 0}$")
    plt.tight_layout()
    plt.savefig("images/thug_hmc_samples_sigma_{}.png".format(
        str(sigma).replace('.', 'dot')), dpi=300)
    plt.show()

