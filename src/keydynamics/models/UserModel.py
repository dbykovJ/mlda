from dataclasses import dataclass
from typing import Self

import numpy as np


@dataclass
class UserModel:
    user_id: int
    samples: np.ndarray
    median: np.ndarray
    average_center_distance: np.ndarray
    threshold: float

    @staticmethod
    def _fit(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        median = np.median(samples, axis=0)  # median per feature (5,)
        # the median absolute deviaton, scalled to estimate the std deviato per feature (5,)
        mad = 1.4826 * np.median(np.abs(samples - median), axis=0) + 1e-8
        return median, mad

    def loo_scores(self) -> np.ndarray:
        return self._loo_scores(self.samples)

    @staticmethod
    def _loo_scores(samples: np.ndarray) -> np.ndarray:
        scores = []
        for i in range(len(samples)):
            median, mad = UserModel._fit(np.delete(samples, i, axis=0))
            scores.append(np.linalg.norm((samples[i] - median) / mad))
        return np.array(scores)

    @classmethod
    def buildModel(
        cls,
        model: np.ndarray,
        user_id: int,
        percentile: float = 85,
        threshold: float | None = None,
    ) -> Self:
        median, mad = cls._fit(model)
        if threshold is None:
            threshold = float(np.percentile(cls._loo_scores(model), percentile))
        return cls(user_id, model, median, mad, threshold)

    def compare_sample(self, sample: np.ndarray) -> float:
        if not self.median or not self.average_center_distance:
            raise ValueError("Model is not fitted yet.")
        return float(
            np.linalg.norm((sample - self.median) / self.average_center_distance)
        )

    def is_user(self, sample: np.ndarray) -> bool:
        return self.compare_sample(sample) <= self.threshold
