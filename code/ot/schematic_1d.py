import numpy as np
import matplotlib.pyplot as plt

# from scipy.stats import wasserstein_distance

def wasserstein_distance(x, y, p, q):
    from scipy.stats.stats import wasserstein_distance
    return np.sqrt(_cdf_distance(2, x, y, p, q))

fig, axes = plt.subplots(1, 3, sharey=True)

x = np.linspace(0, 1, 2000)

p = (x > .1) & (x < .4)
q = (x > .6) & (x < .9)
axes[0].plot(x, p / p.sum(), color='darkred', lw=2)
axes[0].plot(x, q / q.sum(), color='darkblue', lw=2)

d1 = wasserstein_distance(x, x, p, q)

p = (x > .1) & (x < .2)
q = (x > .8) & (x < .9)
axes[1].plot(x, p / p.sum(), color='darkred', lw=2)
axes[1].plot(x, q / q.sum(), color='darkblue', lw=2)

d2 = wasserstein_distance(x, x, p, q)

p = (x > .1) & (x < .85)
q = (x > .15) & (x < .9)
axes[2].plot(x, p / p.sum(), color='darkred', lw=2, label="p(x)")
axes[2].plot(x, q / q.sum(), color='darkblue', lw=2, label="q(x)")

d3 = wasserstein_distance(x, x, p, q)

axes[0].set_ylim(0, axes[0].get_ylim()[1])

for ax in axes:
    ax.set_yticks([])
    ax.set_xticks([0, .5, 1])
    ax.spines["bottom"].set_bounds(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

axes[2].legend()
fig.set_size_inches((6, 3))
fig.tight_layout()
fig.savefig("schematic_1d.png", dpi=400)

axes[0].set_title(
    r"$\mathcal{W}\,$(P, Q) = " + str(np.round(d1, 3)))
axes[1].set_title(
    r"$\mathcal{W}\,$(P, Q) = " + str(np.round(d2, 3)))
axes[2].set_title(
    r"$\mathcal{W}\,$(P, Q) = " + str(np.round(d3, 3)))

fig.subplots_adjust(top=.8)

fig.savefig("schematic_1d_revisited.png", dpi=400)

plt.show()


