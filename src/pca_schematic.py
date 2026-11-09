"""Interactive PCA schematic for the KeyDynamics presentation.

Usage
-----
    uv run scripts/pca_schematic.py        # project environment
    python scripts/pca_schematic.py        # any env with numpy + matplotlib

What it draws
-------------
A synthetic 3-feature point cloud (variance split ~ 50 / 30 / 20 %, rotated
away from the coordinate axes) with its three principal components,
labelled with their explained variance. Schematic companion to the real
5-feature data (dwell, press-press, press-release, release-press,
release-release): same method, 3 features only so the cloud is drawable.

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

    def on_key(event):
        if event.key == "s":
            fig.savefig(SAVE_PATH, dpi=SAVE_DPI)
            print(f"saved {SAVE_PATH} (elev={ax.elev:.0f}, azim={ax.azim:.0f})")

    fig.canvas.mpl_connect("key_press_event", on_key)
    print("drag to rotate | toolbar to zoom/pan | press 's' to save")
    plt.show()


if __name__ == "__main__":
    main()
