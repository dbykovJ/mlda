import numpy as np

from keydynamics.check_sample import check_sample
from keydynamics.model import KeyModel

path = "./out.csv"

model = KeyModel.from_csv(path)

samples = model.to_np()
distances = np.linalg.norm(
    samples[:, None, :] - samples[None, :, :],
    axis=-1,
)

print(distances.mean(), distances.std(), distances.min(), distances.max())


path_emil = "./out_emil.csv"

model_emil = KeyModel.from_csv(path_emil)
sample0 = model_emil.sample_props[0]

rest = check_sample(model, sample0, 20)
print(np.mean([d for _, d in rest]))
