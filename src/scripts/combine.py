"""Merge all model CSVs into one combined CSV, split every keysample into
chunks of MIN_PRESSES..MAX_PRESSES registered presses, and save the resulting
samples as a KeyModel (combined.npy).

Run from the repo root:  .venv/bin/python src/combine.py
Writes combined.csv and combined.npy next to the repo root.
"""

from pathlib import Path

import pandas as pd

from keydynamics.listener.keydata import KeyData
from keydynamics.models.model import KeyModel, SampleProps

CSVS = [
    Path("out.csv"),
    Path("out1.csv"),
    Path("out2.csv"),
    Path("out3.csv"),
    Path("out4.csv"),
    Path("out5.csv"),
    Path("out6.csv"),
    Path("out7.csv"),
    Path("models/lorenz/fixed_text.csv"),
    Path("models/lorenz/coding.csv"),
    Path("models/lorenz/monkeytype.csv"),
]
MIN_PRESSES = 20  # chunks with fewer registered presses are dropped
MAX_PRESSES = 60  # chunks with more registered presses are split further
COMBINED_CSV = Path("lorenz_combined.csv")
OUTPUT = Path("lorenz_combined.npy")


def split_events(events: list[KeyData]) -> list[list[KeyData]]:
    """Split the event list into contiguous chunks whose registered press count
    stays within MIN_PRESSES..MAX_PRESSES. Groups with at most MAX_PRESSES
    presses are returned whole."""
    press_idx = [i for i, e in enumerate(events) if e.event == "PRESS"]
    n_presses = len(press_idx)
    if n_presses <= MAX_PRESSES:
        return [events]
    n_chunks = -(-n_presses // MAX_PRESSES)  # ceil
    cuts = [press_idx[k * n_presses // n_chunks] for k in range(1, n_chunks)]
    bounds = [0, *cuts, len(events)]
    return [events[bounds[i] : bounds[i + 1]] for i in range(n_chunks)]


def main():
    df = pd.concat((pd.read_csv(p) for p in CSVS), ignore_index=True)
    df.to_csv(COMBINED_CSV, index=False)
    print(f"merged {len(CSVS)} csvs -> {COMBINED_CSV} ({len(df)} rows)")

    props = []
    dropped = 0
    for _, group in df.groupby(["session_id", "sample_id"]):
        events = sorted(
            (KeyData(**row) for row in group.to_dict("records")),
            key=lambda e: e.timestamp,
        )
        for chunk in split_events(events):
            presses = sum(1 for e in chunk if e.event == "PRESS")
            if presses < MIN_PRESSES:
                dropped += 1
                continue
            sample = SampleProps.from_sample(chunk)
            if sample is not None:
                props.append(sample)
            else:
                dropped += 1

    model = KeyModel(sample_props=props)
    model.save(str(OUTPUT))
    print(
        f"{len(props)} samples ({dropped} dropped, chunks of "
        f"{MIN_PRESSES}-{MAX_PRESSES} presses) -> {OUTPUT.resolve()}"
    )


if __name__ == "__main__":
    main()
