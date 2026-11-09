"""Model plots: k-NN attribution regions for two imported models, plus a
single-model plot with its UserModel decision contour.

Run from the repo root:  .venv/bin/python src/plot_models.py
Writes voronoi_regions.png and single_model.png next to the repo root.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from keydynamics.models.model import KeyModel
from keydynamics.models.UserModel import UserModel
from keydynamics.ui.projection import ImportedModel, ModelProjection

MODELS = [
    Path("models/dima/dima_casual.npy"),
    Path("models/emil/coding_emil.npy"),
    Path("models/lorenz/coding.npy"),
]
SINGLE_CSV = Path("out4.csv")
TWO_MODELS = [
    Path("emil_large.npy"),
    Path("lorenz.npy"),
]
# Display labels keyed by file name; raw filenames are never used as labels.
MODEL_LABELS = {
    "emil_large.npy": "email-focused",
    "lorenz.npy": "lorenz-focused",
}
VORONOI_OUTPUT = Path("voronoi_regions_k{k}.png")
SINGLE_OUTPUT = Path("single_model.png")
TWO_OUTPUT = Path("two_models.png")
KS = (3, 6, 9)
RESOLUTION = 200


def plot_voronoi_regions():
    projection = ModelProjection()
    for path in MODELS:
        projection.add_model(str(path.resolve()))

    all_points = np.vstack([entry.points for entry in projection.projected])
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    pad = np.maximum((high - low) * 0.2, 0.05)
    extent = (low[0] - pad[0], high[0] + pad[0], low[1] - pad[1], high[1] + pad[1])

    names = {entry.user_id: entry.path.stem for entry in projection.models}
    user_ids = sorted(names)
    palette = ["#3478b8", "#27966f", "#9467bd", "#c18a16", "#298caa", "#9b6b43"]
    colors = [palette[user_ids.index(uid) % len(palette)] for uid in user_ids]
    remap = {uid: i for i, uid in enumerate(user_ids)}

    for k in KS:
        xs, ys, field = projection.global_grid(extent, resolution=RESOLUTION, k=k)
        mapped = np.vectorize(remap.get)(field)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(
            mapped.T,
            origin="lower",
            extent=(xs[0], xs[-1], ys[0], ys[-1]),
            aspect="auto",
            cmap=ListedColormap(colors),
            alpha=0.45,
            interpolation="nearest",
        )
        for entry in projection.projected:
            name = names[entry.source.user_id]
            color = colors[user_ids.index(entry.source.user_id)]
            ax.scatter(
                entry.points[:, 0],
                entry.points[:, 1],
                s=18,
                color=color,
                alpha=0.8,
                label=name,
                edgecolors="none",
            )
            ax.scatter(entry.center[0], entry.center[1], s=120, color=color, marker="x")
        ax.set_xlabel("Principal component 1")
        ax.set_ylabel("Principal component 2")
        ax.set_title(
            f"k-NN attribution regions (k={k}) — {', '.join(names[u] for u in user_ids)}"
        )
        ax.legend()
        fig.tight_layout()
        output = Path(str(VORONOI_OUTPUT).format(k=k))
        fig.savefig(output, dpi=150)
        plt.close(fig)
        print(f"wrote {output.resolve()} (user ids: {np.unique(field).tolist()})")


def plot_single_model():
    model = KeyModel.from_csv(str(SINGLE_CSV))
    samples = model.clean().to_numpy(dtype=float)
    scaler = StandardScaler().fit(samples)
    pca = PCA(n_components=2).fit(scaler.transform(samples))
    points = pca.transform(scaler.transform(samples))
    user_model = UserModel.buildModel(model.to_np(), user_id=1)

    # Reuse the projection's grid machinery for the decision contour.
    projection = ModelProjection()
    projection.scaler = scaler
    projection.pca = pca
    entry = ImportedModel(
        model, samples, SINGLE_CSV.name, user_id=1, user_model=user_model
    )
    extent = projection.decision_extent(
        entry, _padded_extent(points), RESOLUTION
    )
    xs, ys, z = projection.decision_grid(entry, extent, resolution=RESOLUTION)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.contour(
        xs, ys, z.T, levels=[user_model.threshold], colors="#c0392b", linewidths=2
    )
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=18,
        color="#3478b8",
        alpha=0.8,
        label="KeyDynamics Model",
    )
    ax.scatter(
        points.mean(axis=0)[0],
        points.mean(axis=0)[1],
        s=120,
        color="#3478b8",
        marker="x",
    )
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title(
        f"KeyDynamics Model: {len(samples)} samples, decision contour at "
        f"threshold {user_model.threshold:.2f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(SINGLE_OUTPUT, dpi=150)
    print(f"wrote {SINGLE_OUTPUT.resolve()} ({len(samples)} samples)")


def plot_two_user_models():
    projection = ModelProjection()
    for path in TWO_MODELS:
        projection.add_model(str(path.resolve()))

    palette = ["#3478b8", "#27966f", "#9467bd", "#c18a16", "#298caa", "#9b6b43"]
    fig, ax = plt.subplots(figsize=(8, 8))
    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    for index, entry in enumerate(projection.projected):
        color = palette[index % len(palette)]
        label = MODEL_LABELS.get(entry.source.name, entry.source.path.stem)
        extent = projection.decision_extent(
            entry.source, _padded_extent(entry.points), RESOLUTION
        )
        xs, ys, z = projection.decision_grid(entry.source, extent, resolution=RESOLUTION)
        ax.contour(
            xs, ys, z.T, levels=[entry.source.user_model.threshold],
            colors=color, linewidths=2,
        )
        ax.scatter(
            entry.points[:, 0], entry.points[:, 1], s=18,
            color=color, alpha=0.8, label=label, edgecolors="none",
        )
        ax.scatter(entry.center[0], entry.center[1], s=120, color=color, marker="x")
        xmin = min(xmin, xs[0], entry.points[:, 0].min())
        xmax = max(xmax, xs[-1], entry.points[:, 0].max())
        ymin = min(ymin, ys[0], entry.points[:, 1].min())
        ymax = max(ymax, ys[-1], entry.points[:, 1].max())
    margin_x = max((xmax - xmin) * 0.05, 0.05)
    margin_y = max((ymax - ymin) * 0.05, 0.05)
    ax.set_xlim(xmin - margin_x, xmax + margin_x)
    ax.set_ylim(ymin - margin_y, ymax + margin_y)
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    labels = ", ".join(MODEL_LABELS.get(p.name, p.stem) for p in TWO_MODELS)
    ax.set_title(f"User-model decision contours — {labels}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(TWO_OUTPUT, dpi=150)
    print(f"wrote {TWO_OUTPUT.resolve()}")


def _padded_extent(points: np.ndarray):
    low, high = points.min(axis=0), points.max(axis=0)
    pad = np.maximum((high - low) * 0.2, 0.05)
    return (low[0] - pad[0], high[0] + pad[0], low[1] - pad[1], high[1] + pad[1])


def main():
    plot_voronoi_regions()
    plot_single_model()
    plot_two_user_models()


if __name__ == "__main__":
    main()
