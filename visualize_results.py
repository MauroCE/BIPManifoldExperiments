import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Load data
with open("results/experiment1234.pkl", "rb") as file:
    results = pickle.load(file)

with open("results/experiment1234_hmc_prop_parallel.pkl", "rb") as file:
    results_prop = pickle.load(file)

with open("results/experiment1234_thug_prop_parallel_full.pkl", "rb") as file:
    results_prop_thug = pickle.load(file)

sigmas = np.logspace(start=0.0, stop=-6.0, num=7, endpoint=True, base=10.0, dtype=np.float64)
cs = ['dodgerblue', 'lightcoral', 'lawngreen']
mecs = ['navy', 'brown', 'forestgreen']

rc('font', **{'family': 'STIXGeneral'})
fig, ax = plt.subplots()
for ix, alg in enumerate(['thug', 'hmc', 'crwm']):
    ax.plot(sigmas, results['cc-az'][alg],
            marker='o', lw=2.5, ms=9.0, c=cs[ix], mec=mecs[ix], label=alg.upper(), mew=2.0)
ax.plot(sigmas[:4], results_prop['cc-az'][:4], marker='o', lw=2.5, ms=9.0, c='gold', mec='goldenrod',
        label='HMC' + r" $\mathregular{\delta\propto\sigma}$", mew=2.0)
ax.set_ylabel("minESS/s", fontsize=16)
ax.set_xlabel("Noise scale" + r" $\mathregular{\sigma}$", fontsize=16)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
plt.show()
