import numpy as np
import matplotlib.pyplot as plt
import ot
from scipy.spatial.distance import squareform, pdist
from demo import demo_wasserstein

rs = np.random.RandomState(1234)

x = np.linspace(0, 1, 100)

def gpdf(mu, s2):
    return np.exp(-(0.5 / s2) * (x - mu) ** 2 )

p = np.zeros_like(x)
for m in rs.rand(30):
    p += gpdf(m - .2, 0.002)
p /= p.sum()

q = np.zeros_like(x)
for m in rs.rand(30):
    q += gpdf(m + .2, 0.002)
q /= q.sum()

# print(np.sqrt(ot.emd2(p, q, squareform(pdist(x[:, None], metric="sqeuclidean")))))
dist, T = demo_wasserstein(x[:, None], p, q)
# print(dist)

fig = plt.figure()
ax1 = fig.add_axes([.03, .6, .25, .2])
ax2 = fig.add_axes([.03, .25, .25, .2])

ax1.plot(x, p, color="darkred")
ax2.plot(x, q, color="darkblue")
ax1.text(.5, .025, "p(x)", horizontalalignment="center", fontweight="bold", color="darkred")
ax2.text(.5, .02,  "q(x)", horizontalalignment="center", fontweight="bold", color="darkblue")

for ax in (ax1, ax2):
    ax.set_xticks([0, .5, 1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 1)
    ax.set_ylim(0, .025)
    ax.set_yticks([])


ax3 = fig.add_axes([.6, .05, .33, .6])
im3 = ax3.imshow(
    squareform(pdist(x[:, None], metric='sqeuclidean'))[:, ::-1],
    aspect="auto", interpolation="none", origin="lower"
)
xbnds = ax3.get_xlim()
ybnds = ax3.get_ylim()
ax3.xaxis.tick_top()
ax3.set_xticks(np.linspace(xbnds[0], xbnds[1], 3))
ax3.set_yticks(np.linspace(ybnds[0], ybnds[1], 3))
ax3.set_xticklabels(["0.0", "0.5", "1.0"])
ax3.set_yticklabels(["1.0", "0.5", "0.0"])
txt3 = ax3.text(5, x.size - 10, "cost\nmatrix, C", fontweight="bold", color="white", verticalalignment="top")

ax4 = fig.add_axes([.44, .05, .09, .6])
ax4.plot(q, x[::-1], color="darkblue")
ax4.set_xlim(0, .025)
ax4.invert_xaxis()
ax4.set_ylim([0, 1])
ax4.yaxis.tick_right()
ax4.set_yticks([0, .5, 1])
ax4.set_yticklabels([])
ax4.set_xticks([])
ax4.spines["top"].set_visible(False)
ax4.spines["left"].set_visible(False)
ax4.spines["bottom"].set_visible(False)

ax5 = fig.add_axes([.6, .77, .33, .17])
ax5.plot(x, p, color="darkred")
ax5.set_xlim([0, 1])
ax5.set_yticks([])
ax5.set_xticks([0, .5, 1])
ax5.set_xticklabels([])
ax5.spines["top"].set_visible(False)
ax5.spines["left"].set_visible(False)
ax5.spines["right"].set_visible(False)
ax5.set_ylim(0, .025)

ax6 = fig.add_axes([.32, .4, .15, .1])
ax6.arrow(0, 0, 1, 0, lw=0, width=.5, head_width=1, head_length=.5, color="darkgrey")
ax6.set_xlim([0, 2])
ax6.set_ylim([-1, 1])
ax6.axis("off")

fig.set_size_inches((6, 3))
fig.savefig("example_1d_transport_cost.png", dpi=400)

im3.set_array(T[::-1])
im3.set_clim([0, np.percentile(T, 99.9)])
txt3.set_text("optimal\ntransport\nplan, T*")
txt3.set_x(x.size * .6)
txt3.set_y(x.size - 5)
fig.savefig("example_1d_transport_plan.png", dpi=400)

plt.show()