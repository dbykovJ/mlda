import numpy as np
from model import KeyModel, SampleProps


def check_sample(model: KeyModel, sample: SampleProps, k: int):
    sample_np = sample.to_np()

    distances = []

    for i, model_sample in enumerate(model.sample_props):
        distance = np.linalg.norm(sample_np - model_sample.to_np())
        distances.append((i, distance))

    distances.sort(key=lambda x: x[1])
    return distances[:k], model.sample_props[distances[0][0]:distances[k][0]]

if __name__ == "__main__":

    s1 = SampleProps(dwell_time=1, press_press=1, press_release=1, release_press=1, release_release=1)
    s2 = SampleProps(dwell_time=2, press_press=2, press_release=2, release_press=2, release_release=2)
    s3 = SampleProps(dwell_time=3, press_press=3, press_release=3, release_press=3, release_release=3)
    model = KeyModel(sample_props=[s1, s2, s3])

    sample = SampleProps(dwell_time=1, press_press=1, press_release=1, release_press=1, release_release=1)
    k = 2
    
    nearest_neighbors, neighbor_samples = check_sample(model, sample, k)

    print("Nearest Neighbors (index, distance):")
    for index, distance in nearest_neighbors:
        print(f"Index: {index}, Distance: {distance}")
