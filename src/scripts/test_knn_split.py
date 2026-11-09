"""k-NN vote error vs k, with large keysamples split into chunks of at most
SPLIT_THRESHOLD presses and an 80/20 train/test split per model.

Run from the repo root:  .venv/bin/python src/test_knn_split.py
Writes knn_split.png next to the repo root.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keydynamics.listener.keydata import KeyData
from keydynamics.models.GlobalModel import GlobalModel
from keydynamics.models.model import SampleProps

MODEL_CSVS = [
    Path("models/dima/fixed_text.csv"),
    Path("models/emil/fixed_text.csv"),
    Path("models/lorenz/fixed_text.csv"),
]
SPLIT_THRESHOLD = 50  # keysamples with more registered presses are split in two
TRAIN_FRACTION = 0.8
KS = range(1, 21)
OUTPUT = Path("knn_split.png")
SEED = 0


def split_events(events: list[KeyData]) -> list[list[KeyData]]:
    """Recursively halve the event list until every chunk has at most
    SPLIT_THRESHOLD registered presses."""
    presses = sum(1 for e in events if e.event == "PRESS")
    if presses <= SPLIT_THRESHOLD:
        return [events]
    mid = len(events) // 2
    return split_events(events[:mid]) + split_events(events[mid:])


def load_samples(path: Path) -> tuple[list[tuple[str, str, SampleProps]], int]:
    """Read a key-event CSV and build samples grouped by (session_id, sample_id).
    Any keysample with more than SPLIT_THRESHOLD registered keypresses is split
    into chunks of at most SPLIT_THRESHOLD presses; each chunk gets its own
    derived sample_id ("<id>.0", "<id>.1", ...).
    Returns (session_id, sample_id, props) tuples and the number of dropped chunks."""
    df = pd.read_csv(path)
    samples = []
    dropped = 0
    for _, group in df.groupby(["session_id", "sample_id"]):
        session_id = str(group["session_id"].iloc[0])
        sample_id = group["sample_id"].iloc[0]
        events = sorted(
            (KeyData(**row) for row in group.to_dict("records")),
            key=lambda e: e.timestamp,
        )
        chunks = split_events(events)
        for i, chunk in enumerate(chunks):
            new_id = str(sample_id) if len(chunks) == 1 else f"{sample_id}.{i}"
            sample = SampleProps.from_sample(chunk)
            if sample is not None:
                samples.append((session_id, new_id, sample))
            else:
                dropped += 1
    return samples, dropped


def main():
    rng = np.random.default_rng(SEED)
    names = [path.parent.name for path in MODEL_CSVS]
    train_sets = []  # (user_id, name, train features)
    test_sets = []  # (user_id, name, test features)
    for user_id, path in enumerate(MODEL_CSVS, start=1):
        samples, dropped = load_samples(path)
        features = np.array([p.to_np() for _, _, p in samples])
        order = rng.permutation(len(features))
        cut = int(len(features) * TRAIN_FRACTION)
        train_sets.append((user_id, path.parent.name, features[order[:cut]]))
        test_sets.append((user_id, path.parent.name, features[order[cut:]]))
        print(
            f"{path.parent.name}: {len(features)} samples "
            f"({dropped} dropped, split into <= {SPLIT_THRESHOLD}-press chunks), "
            f"train {len(order[:cut])} / test {len(order[cut:])}"
        )

    # One global model per training set, plus their combination for voting.
    per_model = [GlobalModel([(x, uid) for x in train]) for uid, _, train in train_sets]
    combined = GlobalModel()
    for gm in per_model:
        combined.samples.extend(gm.samples)

    all_test = [(x, uid) for uid, _, test in test_sets for x in test]
    combined_mse = []
    per_model_mse = {name: [] for _, name, _ in test_sets}
    for k in KS:
        errors = np.array(
            [(combined.vote(x, k) - uid) ** 2 for x, uid in all_test], dtype=float
        )
        combined_mse.append(errors.mean())
        for uid, name, test in test_sets:
            sub = np.array(
                [(combined.vote(x, k) - uid) ** 2 for x in test], dtype=float
            )
            per_model_mse[name].append(sub.mean())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(KS, combined_mse, "k-o", label="combined", linewidth=2)
    for (uid, name, _), gm in zip(train_sets, per_model):
        ax.plot(KS, per_model_mse[name], "--", label=f"{name} (test only)")
    ax.set_xlabel("k")
    ax.set_ylabel("mean squared error (user_id)")
    ax.set_title(
        f"k-NN vote error vs k — {len(combined.samples)} train samples, "
        f"{len(all_test)} test samples"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150)
    print(f"wrote {OUTPUT.resolve()}")
    best = int(KS[int(np.argmin(combined_mse))])
    print(f"best k by combined MSE: {best} (mse {min(combined_mse):.3f})")


if __name__ == "__main__":
    main()
