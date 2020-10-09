import numpy as np
import matplotlib.pyplot as plt

rs = np.random.RandomState(1234)
fig, axes = plt.subplots(1, 3, sharey=True)

x = np.linspace(-1, 4, 1000)

def gpdf(mu, s2):
    return np.exp(-(0.5 / s2) * (x - mu) ** 2 )

p = np.zeros_like(x)
for m in rs.rand(30):
    p += gpdf(m, 0.003)
p /= p.sum()

q = np.zeros_like(x)
for m in rs.rand(30):
    q += gpdf(2 + m, 0.003)
q /= q.sum()

axes[0].plot(x, p, color='darkred', lw=2, label='p(x)')
axes[0].plot(x, q, color='darkblue', lw=2, label='q(x)')
axes[0].set_yticks([-.01, 0, .01])
axes[0].set_ylim([-.01, .01])
axes[0].legend(loc="lower right")

axes[1].fill_between(x, p, color='#7C4700', lw=2)
axes[1].fill_between(x, -q, color='#281700', lw=2)
axes[1].arrow(
    1.3, .002, .8, 0,
    lw=0, width=.0005,
    color="midnightblue", head_width=0.001, head_length=.3
)

axes[2].fill_between(x, q, color='#7C4700', lw=2, label="dirt")
axes[2].fill_between(x, -p, color='#281700', lw=2, label="hole")
axes[2].arrow(
    1.6, .002, -.8, 0,
    lw=0, width=.0005,
    color="midnightblue", head_width=0.001, head_length=.3
)
axes[2].legend(loc="lower right")

axes[0].set_title("density functions")
axes[1].set_title(r"transport $P \rightarrow Q$")
axes[2].set_title(r"transport $Q \rightarrow P$")

for ax in axes:
    ax.set_xticks([])

fig.set_size_inches((8, 2.5))
fig.tight_layout()

fig.savefig("symmetry_1d.png", dpi=400)
plt.show()
