from collections import Counter
from dataclasses import dataclass, field
import numpy as np

@dataclass
class GlobalModel:
    samples: list[tuple[np.ndarray, int]] = field(default_factory=list)


    def check_sample(self, sample: np.ndarray, k: int) -> float:
        distances = []
        for model_sample in self.samples:
            distance = np.linalg.norm(sample - model_sample[0])
            distances.append([distance, model_sample[1]])


        distances.sort(key=lambda x: x[0])
        counts = Counter(s[1] for s in distances[:k])
        user_id, count = counts.most_common(1)[0]
        self.samples.append((sample, user_id))
        return user_id


    def add_sample(self, sample: np.ndarray, user_id: int) -> None:
        self.samples.append((sample, user_id))



globalModel = GlobalModel()