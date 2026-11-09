"""Interactive PCA schematic for the KeyDynamics presentation.

Usage
-----
    uv run src/pca_schematic.py            # project environment
    python src/pca_schematic.py            # any env with numpy + matplotlib
    python src/pca_schematic.py --save     # headless: render + save, no GUI

What it draws
-------------
A synthetic 3-feature point cloud (variance split ~ 50 / 30 / 20 %, rotated
away from the coordinate axes) with its three principal components,
labelled with their explained variance. One exemplary point x is picked
and projected into the plane spanned by v1 and v2:

- the plane, drawn translucently,
- the projected point x~ = z1 v1 + z2 v2, with thin parallelogram sides
  showing the linear combination of the two components,
- a dashed line connecting x and x~ -- by construction parallel to v3,
  i.e. the discarded part z3 v3 of the point.

Schematic companion to the real 5-feature data (dwell, press-press,
press-release, release-press, release-release): same method, 3 features
only so the cloud is drawable.

Controls
--------
- Drag with the mouse: rotate the view (arcball style).
- Toolbar magnifier or right-drag: zoom / pan.
- Press 's' in the figure window: save 'pca_cloud.png' (200 dpi) and print
  the current elevation/azimuth so the view can be reproduced.
  The toolbar save button works as well.

Tweak the constants below to taste; nothing else needs changing.
"""

from __future__ import annotations

import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

# --------------------------------------------------------------------- settings
SEED = 42  # reproducible cloud
N_SAMPLES = 160
STD_ALONG_PCS = (3.0, 2.32, 1.90)  # -> explained variance ~ 50/30/20 %
ROT_Z_DEG = 28.0  # rotate cloud off the coordinate axes
ROT_X_DEG = -18.0
ELEV, AZIM = 18, -173  # start view: all three PCs separated
ZOOM = 1.3  # overall size of the 3D box in the figure
POINT_SIZE = 24
POINT_COLOR = "#4a7bb7"
PC_COLORS = ("#d1495b", "#2e933c", "#7a4fbf")
EXAMPLE_INDEX = None  # None: auto-pick (see below)
EXAMPLE_COLOR = "#e07a1f"
PLANE_COLOR = "#8fb3d9"
PLANE_ALPHA = 0.16
FIG_SIZE = (6.5, 7.2)
SAVE_PATH = "pca_cloud.png"
SAVE_DPI = 200


def rot_z(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def make_cloud():
    rng = np.random.default_rng(SEED)
    R = rot_z(np.deg2rad(ROT_Z_DEG)) @ rot_x(np.deg2rad(ROT_X_DEG))
    return rng.standard_normal((N_SAMPLES, 3)) * np.array(STD_ALONG_PCS) @ R.T


def pca(X):
    C = np.cov(X, rowvar=False)
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    return evals[order], evecs[:, order]


def main():
    save_only = "--save" in sys.argv
    if save_only:
        matplotlib.use("Agg")

    X = make_cloud()
    evals, evecs = pca(X)
    share = evals / evals.sum()
    print("explained variance:", np.round(share, 3))

    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(projection="3d")

    ax.scatter(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        s=POINT_SIZE,
        c=POINT_COLOR,
        alpha=0.55,
        depthshade=True,
        edgecolors="none",
    )

    for i, color in enumerate(PC_COLORS):
        v = evecs[:, i]
        length = 2.0 * np.sqrt(evals[i])  # arrow length = 2 std devs
        ax.quiver(
            0, 0, 0, *(length * v), color=color, linewidth=2.2, arrow_length_ratio=0.11
        )
        tip = (length + 0.9) * v
        ax.text(
            tip[0],
            tip[1],
            tip[2],
            rf"$\mathbf{{v}}_{i + 1}$"
            + "\n"
            + rf"$\lambda_{i + 1} = {100 * share[i]:.1f}\,\%$",
            color=color,
            fontsize=11.5,
            fontweight="bold",
            ha="center",
            va="center",
        )

    lim = 1.0 * np.abs(X).max()  # frame contains every point
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1), zoom=ZOOM)  # equal scales + fill level
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_zlabel("feature 3")
    ax.set_title(
        "Point cloud with its principal components\n"
        "schematic \u00b7 3 features \u00b7 synthetic data",
        fontsize=12.5,
    )
    ax.grid(False)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(NullLocator())  # schematic: no numeric ticks
    ax.view_init(elev=ELEV, azim=AZIM)

    # ------------------------------------------- plane + projected example point
    v1, v2, v3 = evecs[:, 0], evecs[:, 1], evecs[:, 2]
    scores = X @ evecs  # (n, 3): z1, z2, z3 of every sample
    if EXAMPLE_INDEX is not None:
        idx = int(EXAMPLE_INDEX)
    else:
        # best-projecting sample in the positive z1/z2 quadrant: among the
        # samples whose distance from the plane is clearly visible but
        # not dominant, take the one that connects most visibly to BOTH
        # axes (maximize min(z1, z2))
        in_plane = np.hypot(scores[:, 0], scores[:, 1])
        eligible = (
            (np.abs(scores[:, 2]) >= 1.0)
            & (scores[:, 0] > 0)
            & (scores[:, 1] > 0)
            & (np.abs(scores[:, 2]) <= 0.75 * in_plane)
        )
        idx = int(np.argmax(np.where(eligible, np.minimum(scores[:, 0], scores[:, 1]), -np.inf)))
    x_star = X[idx]
    z1, z2, z3 = scores[idx]
    x_tilde = z1 * v1 + z2 * v2  # projection of x into the plane
    s3 = np.sign(z3) if z3 != 0 else 1.0

    print(f"example point: z1={z1:.2f}, z2={z2:.2f}, z3={z3:.2f}")
    print(f"discarded when reducing to 2D: z3^2 = {z3**2:.2f}")

    # the plane spanned by v1, v2 (translucent), sized to contain x_tilde
    half = max(0.55 * lim, abs(z1) + 1.0, abs(z2) + 1.0)
    u, w = np.meshgrid(
        np.linspace(-half, half, 2), np.linspace(-half, half, 2)
    )
    plane = u[..., None] * v1 + w[..., None] * v2
    ax.plot_surface(
        plane[..., 0],
        plane[..., 1],
        plane[..., 2],
        color=PLANE_COLOR,
        alpha=PLANE_ALPHA,
        linewidth=0,
        antialiased=True,
    )

    # dashed drop line x -> x~ (parallel to v3: the discarded z3 v3)
    ax.plot(
        [x_star[0], x_tilde[0]],
        [x_star[1], x_tilde[1]],
        [x_star[2], x_tilde[2]],
        linestyle=(0, (4, 3)),
        color="#444444",
        linewidth=1.4,
    )
    drop_mid = 0.5 * (x_star + x_tilde) + 0.3 * s3 * v3
    ax.text(
        *drop_mid,
        r"$z_3\,\mathbf{v}_3$" + "\n(discarded)",
        color="#444444",
        fontsize=9,
        ha="center",
        va="center",
    )

    # thin parallelogram sides: the linear combination z1 v1 + z2 v2
    p1, p2 = z1 * v1, z2 * v2
    ax.plot(
        [p1[0], x_tilde[0]], [p1[1], x_tilde[1]], [p1[2], x_tilde[2]],
        color=PC_COLORS[1], alpha=0.6, linewidth=1.8,
    )
    ax.plot(
        [p2[0], x_tilde[0]], [p2[1], x_tilde[1]], [p2[2], x_tilde[2]],
        color=PC_COLORS[0], alpha=0.6, linewidth=1.8,
    )

    # the example point and its projection
    ax.scatter(
        *x_star, s=70, c=EXAMPLE_COLOR, edgecolors="black",
        linewidths=0.5, depthshade=False,
    )
    ax.scatter(
        *x_tilde, s=55, c=EXAMPLE_COLOR, marker="D", edgecolors="black",
        linewidths=0.5, depthshade=False,
    )
    ax.text(
        *(x_star + 0.45 * s3 * v3),
        r"$\mathbf{x}$",
        color=EXAMPLE_COLOR,
        fontsize=12,
        fontweight="bold",
        ha="center",
        va="center",
    )
    ax.text(
        *(x_tilde - 0.55 * s3 * v3),
        r"$\tilde{\mathbf{x}} = z_1\mathbf{v}_1 + z_2\mathbf{v}_2$",
        color="#444444",
        fontsize=9,
        ha="center",
        va="center",
    )

    if save_only:
        fig.savefig(SAVE_PATH, dpi=SAVE_DPI)
        print(f"saved {SAVE_PATH}")
        plt.close(fig)
        return

    def on_key(event):
        if event.key == "s":
            fig.savefig(SAVE_PATH, dpi=SAVE_DPI)
            print(f"saved {SAVE_PATH} (elev={ax.elev:.0f}, azim={ax.azim:.0f})")

    fig.canvas.mpl_connect("key_press_event", on_key)
    print("drag to rotate | toolbar to zoom/pan | press 's' to save")
    plt.show()


if __name__ == "__main__":
    main()
