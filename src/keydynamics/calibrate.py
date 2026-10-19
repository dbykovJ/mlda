import numpy as np

from keydynamics.model import KeyModel

path = "./out_test.csv"

model = KeyModel.from_csv(path)

samples = model.to_np()
distances = np.linalg.norm(
    samples[:, None, :] - samples[None, :, :],
    axis=-1,
)

print(distances.mean(), distances.std(), distances.min(), distances.max())