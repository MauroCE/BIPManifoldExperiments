import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Load data
# B = 10
with open("results/experiment1234_final_T10_step005.pkl", "rb") as file:
    results10_005 = pickle.load(file)

with open("results/experiment1234_final_T10_step01.pkl", "rb") as file:
    results10_01 = pickle.load(file)

with open("results/experiment1234_final_T10_step05.pkl", "rb") as file:
    results10_05 = pickle.load(file)

# B = 20
with open("results/experiment1234_final_T20_step005.pkl", "rb") as file:
    results20_005 = pickle.load(file)

with open("results/experiment1234_final_T20_step01.pkl", "rb") as file:
    results20_01 = pickle.load(file)

with open("results/experiment1234_final_T20_step05.pkl", "rb") as file:
    results20_05 = pickle.load(file)

# B = 30
with open("results/experiment1234_final_T30_step005.pkl", "rb") as file:
    results30_005 = pickle.load(file)

with open("results/experiment1234_final_T30_step01.pkl", "rb") as file:
    results30_01 = pickle.load(file)

with open("results/experiment1234_final_T30_step05.pkl", "rb") as file:
    results30_05 = pickle.load(file)

results = {
    '10': [results10_005, results10_01, results10_05],
    '20': [results20_005, results20_01, results20_05],
    '30': [results30_005, results30_01, results30_05]
}

sigmas = np.logspace(start=0.0, stop=-5.0, num=6, endpoint=True, base=10.0, dtype=np.float64)
cs = ['dodgerblue', 'lightcoral', 'lawngreen']
mecs = ['navy', 'brown', 'forestgreen']
algs = ['thug', 'hmc', 'crwm']
Bs = [10, 20, 30]
steps = [0.05, 0.1, 0.5]

rc('font', **{'family': 'STIXGeneral'})
fig, ax = plt.subplots(figsize=(12, 10), nrows=3, ncols=3, sharex=True, sharey=True)

for ri in range(3):      # row index (for step sizes)
    for ci in range(3):  # col index (for number of integration steps)
        for ai, alg in enumerate(algs):
            ax[ri, ci].plot(sigmas, results[str(10*(ci+1))][ri]['cc-az'][alg],
                            marker='o', lw=2.5, ms=9.0, c=cs[ai], mec=mecs[ai], label=alg.upper(), mew=2.0)
        ax[ri, ci].plot(sigmas[:4], results[str(10*(ci+1))][ri]['cc-az']['hmc-p'][:4],
                        marker='o', lw=2.5, ms=9.0, c='gold', mec='goldenrod',
                        label='HMC' + r" $\mathregular{\delta\propto\sigma}$", mew=2.0)
        ax[ri, ci].set_xscale('log')
        ax[ri, ci].set_yscale('log')
        ax[ri, ci].grid(True, color='gainsboro')
    # prettify
    ax[ri, 0].set_ylabel("minESS/s", fontsize=16)
for ci in range(3):
    ax[0, ci].set_title(r"$\mathregular{B=" + str(Bs[ci]) + "}$", fontsize=18)
    ax[-1, ci].set_xlabel("Noise scale" + r" $\mathregular{\sigma}$", fontsize=16)
plt.legend()
plt.tight_layout()
# plt.savefig("images/min_ess_vs_noise_scale_grid.png", dpi=300)
plt.show()

# Generate a single plot for B=20 and step=0.1 which will be used in the main part of the thesis
rc('font', **{'family': 'STIXGeneral'})
fig, ax = plt.subplots()
for ai, alg in enumerate(algs):
    ax.plot(sigmas, results['20'][1]['cc-az'][alg], marker='o', lw=2.5, ms=9.0, c=cs[ai], mec=mecs[ai],
            label=alg.upper(), mew=2.0)
ax.plot(sigmas[:4], results['20'][1]['cc-az']['hmc-p'][:4], marker='o', lw=2.5, ms=9.0, c='gold', mec='goldenrod',
        label='HMC' + r" $\mathregular{\delta\propto\sigma}$", mew=2.0)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, color='gainsboro')
# Prettify
ax.set_ylabel("minESS/s", fontsize=16)
ax.set_xlabel("Noise scale" + r" $\mathregular{\sigma}$", fontsize=16)
plt.legend()
plt.tight_layout()
# plt.savefig("images/min_ess_vs_noise_scale_main.png", dpi=300)
plt.show()

# Generate raw minESS plot for main part of the thesis
rc('font', **{'family': 'STIXGeneral'})
fig, ax = plt.subplots()
for ai, alg in enumerate(algs):
    ax.plot(sigmas, results['20'][1]['ess-az'][alg], marker='o', lw=2.5, ms=9.0, c=cs[ai], mec=mecs[ai],
            label=alg.upper(), mew=2.0)
ax.plot(sigmas[:4], results['20'][1]['cc-az']['hmc-p'][:4], marker='o', lw=2.5, ms=9.0, c='gold', mec='goldenrod',
        label='HMC' + r" $\mathregular{\delta\propto\sigma}$", mew=2.0)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid(True, color='gainsboro')
# Prettify
ax.set_ylabel("minESS", fontsize=16)
ax.set_xlabel("Noise scale" + r" $\mathregular{\sigma}$", fontsize=16)
plt.legend()
plt.tight_layout()
# plt.savefig("images/raw_min_ess_main.png", dpi=300)
plt.show()


# Join image
rc('font', **{'family': 'STIXGeneral'})
fig, ax = plt.subplots(figsize=(8, 4), ncols=2)
for ai, alg in enumerate(algs):
    ax[0].plot(sigmas, results['20'][1]['cc-az'][alg], marker='o', lw=2.5, ms=9.0, c=cs[ai], mec=mecs[ai],
               label=alg.upper(), mew=2.0)
    ax[1].plot(sigmas, results['20'][1]['ess-az'][alg], marker='o', lw=2.5, ms=9.0, c=cs[ai], mec=mecs[ai],
               label=alg.upper(), mew=2.0)
ax[0].plot(sigmas[:4], results['20'][1]['cc-az']['hmc-p'][:4], marker='o', lw=2.5, ms=9.0, c='gold', mec='goldenrod',
           label='HMC' + r" $\mathregular{\delta\propto\sigma}$", mew=2.0)
ax[1].plot(sigmas[:4], results['20'][1]['ess-az']['hmc-p'][:4], marker='o', lw=2.5, ms=9.0, c='gold', mec='goldenrod',
           label='HMC' + r" $\mathregular{\delta\propto\sigma}$", mew=2.0)
for i in range(2):
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].grid(True, color='gainsboro')
    ax[i].set_xlabel("Noise scale" + r" $\mathregular{\sigma}$", fontsize=16)

# Prettify
ax[0].set_ylabel("minESS/s", fontsize=16)
ax[1].set_ylabel("minESS", fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig("images/comp_cost_vs_noise_scale_and_raw_min_ess_main.png", dpi=400)
plt.show()
