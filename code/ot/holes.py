import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from scipy.stats import multivariate_normal
from scipy.spatial.distance import pdist, squareform
from demo import demo_wasserstein
from scipy.ndimage import gaussian_filter

plt.close("all")

# Dirt piles.
ds = [
    multivariate_normal(
        mean=[-4, -4],
        cov=np.diag([.5, 1])
    ),
    multivariate_normal(
        mean=[0, 4],
        cov=2*np.diag([.5, 1])
    ),
    multivariate_normal(
        mean=[0, 4],
        cov=2*np.diag([.5, 1])
    ),
    multivariate_normal(
        mean=[4, -4],
        cov=3*np.diag([.5, 1])
    ),
    multivariate_normal(
        mean=[4, -4],
        cov=3*np.diag([.5, 1])
    ),
    multivariate_normal(
        mean=[4, -4],
        cov=3*np.diag([.5, 1])
    )
]

# Holes.
hs = [
    multivariate_normal(
        mean=[-5.8, 8.2],
        cov=0.2*np.diag([.5, 1])
    ),
    multivariate_normal(
        mean=[-0.3, -1.7],
        cov=0.2*np.diag([.5, 1])
    ),
    multivariate_normal(
        mean=[-10. ,   7.9],
        cov=0.2*np.diag([.5, 1])
    ),
    multivariate_normal(
        mean=[-8,  -8],
        cov=0.2*np.diag([.5, 1])
    ),
    multivariate_normal(
        mean=[7,  8],
        cov=0.2*np.diag([.5, 1])
    ),
    multivariate_normal(
        mean=[9,  4],
        cov=0.2*np.diag([.5, 1])
    ),
    multivariate_normal(
        mean=[9,  -9],
        cov=0.2*np.diag([.5, 1])
    ),
    multivariate_normal(
        mean=[-10,  0],
        cov=0.2*np.diag([.5, 1])
    )
]

def plot_holes(savefig=True):

    nx, ny = 2000, 1000
    xx, yy = np.meshgrid(
        np.linspace(-13, 20, nx),
        np.linspace(-13, 13, ny),
    )
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    H = np.sum([hd.pdf(xy).reshape((ny, nx)) for hd in hs], axis=0)
    D = np.sum([dd.pdf(xy).reshape((ny, nx)) for dd in ds], axis=0)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(
        D - H,
        cmap="seismic",
        origin="lower",
        clim=(-1.1*D.max(), 1.1*D.max()))

    ax.axis("off")
    x0 = 1550
    y0 = 450
    rx, ry = 35, 35
    rw, rh = 100, 50

    bw = 2*rx + rw + 250
    bh = 3*ry + 2*rh

    ax.add_collection(PatchCollection(
        [Rectangle((x0 + rx, y0 + ry), rw, rh)], zorder=2, facecolor="#0000B6"
    ))
    ax.add_collection(PatchCollection(
        [Rectangle((x0 + rx, y0 + 2*ry + rh), rw, rh)], zorder=2, facecolor="#D50000"
    ))
    ax.add_collection(PatchCollection(
        [Rectangle((x0, y0), bw, bh)],
        zorder=2, facecolor="none", linewidth=1.5, edgecolor="k", alpha=.5
    ))
    ax.text(x0 + 2*rx + rw, y0 + ry + .5*rh, "hole", verticalalignment="center")
    ax.text(x0 + 2*rx + rw, y0 + 2*ry + 1.5*rh, "dirt pile", verticalalignment="center")

    fig.set_size_inches((6, 3))
    if savefig:
        fig.savefig("holes.png", dpi=400)
    return fig, ax, nx, ny

def plot_holes_with_arrow(savefig=True):
    fig, ax, nx, ny = plot_holes(savefig=False)
    ax.arrow(
        nx*.53, ny*.44, 220, 200, zorder=2, linewidth=2,
        head_width=30, head_length=30, color="k")
    ax.text(nx*.53, ny*.47, "(x0, y0)", horizontalalignment="right")
    ax.text(nx*.53 + 220, ny*.46 + 200, "(x1, y1)", horizontalalignment="right")
    ax.text(nx*.53 + 130, ny*.46 + 45, "w", fontweight="bold")
    
    if savefig:
        fig.savefig("holes_arrow.png", dpi=400)
    return fig, ax

def plot_holes_with_grid(savefig=True):
    fig, ax, nx, ny = plot_holes(savefig=False)
    for _x in np.linspace(0, 1400, 12*2):
        ax.axvline(_x, color="k", dashes=[2, 2])
    for _y in np.linspace(0, 995, 10*2):
        ax.plot([0, 1400], [_y, _y], color="k", dashes=[2, 2])
    if savefig:
        fig.savefig("grid_holes.png", dpi=400)
    return fig, ax


def plot_discretized_holes(savefig=True):

    nx, ny = 20, 20
    xx, yy = np.meshgrid(
        np.linspace(-13, 13, nx),
        np.linspace(-13, 13, ny),
    )
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    H = np.sum([hd.pdf(xy).reshape((ny, nx)) for hd in hs], axis=0)
    D = np.sum([dd.pdf(xy).reshape((ny, nx)) for dd in ds], axis=0)
    D /= np.sum(D)
    H /= np.sum(H)
    D = np.clip(D, 1e-5, np.inf)
    H = np.clip(H, 1e-5, np.inf)
    D /= np.sum(D)
    H /= np.sum(H)

    fig = plt.figure()
    ax1 = fig.add_axes([.1, .55, .225, .39])
    ax2 = fig.add_axes([.1, .1, .225, .39])
    ax1.imshow(
        D, cmap="seismic", origin="lower", clim=(-1.1*D.max(), 1.1*D.max()), aspect="auto"
    )
    ax2.imshow(
        -H, cmap="seismic", origin="lower", clim=(-1.1*D.max(), 1.1*D.max()), aspect="auto"
    )
    for ax in (ax1, ax2):
        ax.set_xticks([0, 10, 20])
        ax.set_yticks([0, 10, 20])
    ax1.text(
        3.5, 18.5, "p(x)", color="darkred",
        fontweight="bold", horizontalalignment="center", verticalalignment="top"
    )
    ax1.set_xticklabels([])
    ax2.text(
        10, 18.5, "q(x)", color="darkblue",
        fontweight="bold", horizontalalignment="center", verticalalignment="top"
    )

    ax3 = fig.add_axes([.6, .05, .33, .6])
    im3 = ax3.imshow(squareform(pdist(xy)), aspect="auto", interpolation="none")
    ax3.set_xticks(np.arange(0, 401, 100))
    ax3.set_yticks(np.arange(0, 401, 100))
    ax3.xaxis.tick_top()
    txt3 = ax3.text(15, 15, "cost", verticalalignment="top", color="white", fontweight="bold")
    
    ax4 = fig.add_axes([.44, .05, .09, .6])
    ax4.plot(H.ravel(), np.arange(H.size), color="darkblue")
    ax4.set_ylim([0, H.size])
    ax4.set_yticks(np.arange(0, 401, 100))
    ax4.set_yticklabels([])
    ax4.set_xticklabels([])
    ax4.yaxis.tick_right()
    ax4.invert_xaxis()

    ax5 = fig.add_axes([.6, .77, .33, .17])
    ax5.plot(np.arange(D.size), D.ravel(), color="darkred")
    ax5.set_xlim([0, D.size])
    ax5.set_xticks(np.arange(0, 401, 100))
    ax5.set_xticklabels([])
    ax5.set_yticklabels([])

    ax6 = fig.add_axes([.35, .48, .1, .1])
    ax6.arrow(0, 0, 1, 0, lw=0, width=.5, head_width=1, head_length=.5, color="darkgrey")
    ax6.set_xlim([0, 2])
    ax6.set_ylim([-1, 1])
    ax6.axis("off")

    fig.set_size_inches((6, 3))
    if savefig:
        fig.savefig("discretized_holes_cost.png", dpi=400)

    _, T = demo_wasserstein(xy, D.ravel(), H.ravel())
    print(np.linalg.norm(T.sum(axis=0) - D.ravel()))
    print(np.linalg.norm(T.sum(axis=1) - H.ravel()))

    # np.save("T.npy", T)
    # T = np.load("T.npy")

    Ts = gaussian_filter(T[::-1], sigma=.5)
    im3.set_array(Ts)
    im3.set_clim([0, np.percentile(T, 99.9)])
    txt3.set_text("transport plan")
    if savefig:
        fig.savefig("discretized_holes_transport.png", dpi=400)

    thres = np.percentile(T, 100 * ((T.size - 80) / T.size))
    destination, origin = np.where(T > thres)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(
        (D - H), cmap="seismic", interpolation="None", origin="lower",
        clim=(-1.1*D.max(), 1.1*D.max()), aspect="auto"
    )
    for o, d in zip(origin, destination):
        j0, i0 = np.unravel_index(o, (nx, ny))
        j1, i1 = np.unravel_index(d, (nx, ny))
        ax.arrow(
            i0, j0, i1 - i0, j1 - j0,
            lw=0, width=.075, head_width=0.3, head_length=0.3, color="Gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Optimal Transport Solution!")
    fig.set_size_inches((4, 3))
    fig.tight_layout()
    if savefig:
        fig.savefig("holes_arrows.png", dpi=400)


if __name__ == "__main__":
    plot_holes()
    plot_holes_with_arrow()
    plot_holes_with_grid()
    plot_discretized_holes()
    plt.show()