import numpy as np

from keydynamics.model import KeyModel, SampleProps


def check_sample(
    model: KeyModel, sample: SampleProps, k: int
) -> list[tuple[int, float]]:
    sample_np = sample.to_np()

    distances: list[tuple[int, float]] = []

    for i, model_sample in enumerate(model.sample_props):
        distance = np.linalg.norm(sample_np - model_sample.to_np())
        distances.append((i, float(distance)))

    distances.sort(key=lambda x: x[1])
    return distances[:k]
