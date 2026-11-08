from dataclasses import dataclass

import numpy as np

from keydynamics.models import UserModel
from keydynamics.models.GlobalModel import GlobalModel


@dataclass
class ModelController:
    global_model: GlobalModel
    user_models: list[UserModel]

    def add_user_model(self, user_model: UserModel) -> None:
        self.user_models.append(user_model)

    def define_global_model(self) -> None:
        self.global_model = GlobalModel()
        for user_model in self.user_models:
            for sample in user_model.samples:
                self.global_model.add_sample(sample, user_model.user_id)


    def check_sample_globally(self, sample: np.ndarray, k: int) -> int:
        return self.global_model.check_sample(sample, k)


    def check_sample_per_user(self, sample: np.ndarray) -> list[tuple[int, bool]]:
        results = []
        for user_model in self.user_models:
            is_user = user_model.is_user(sample)
            results.append((user_model.user_id, is_user))
        return results


    def clear_models(self) -> None:
        self.global_model.samples.clear()
        self.user_models.clear()
