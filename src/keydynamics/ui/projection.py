"""Model loading and shared PCA calculations, independent of Qt."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from keydynamics.models.model import KeyModel
from keydynamics.models.UserModel import UserModel


@dataclass
class ImportedModel:
    model: KeyModel
    samples: np.ndarray
    name: str
    path: Path | None = None
    user_id: int = 0
    user_model: UserModel | None = None


@dataclass
class ProjectedModel:
    source: ImportedModel
    points: np.ndarray
    center: np.ndarray


class ModelProjection:
    def __init__(self):
        self.models: list[ImportedModel] = []
        self.projected: list[ProjectedModel] = []
        self.scaler = None
        self.pca = None
        self._next_user_id = 1

    @staticmethod
    def _clean_samples(model: KeyModel) -> np.ndarray:
        if not model.sample_props:
            raise ValueError("The model contains no samples.")
        if not np.isfinite(model.to_np()).all():
            raise ValueError("Model features must be finite numbers.")
        samples = model.clean().to_numpy(dtype=float)
        if len(samples) < 2 or not np.isfinite(samples).all():
            raise ValueError("At least two finite samples must remain after cleaning.")
        return samples

    def _refit(self, models: list[ImportedModel]) -> None:
        combined = np.vstack([entry.samples for entry in models])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(combined)
        if not np.any(np.var(scaled, axis=0) > 0):
            raise ValueError("The samples have no variation to visualize.")
        pca = PCA(n_components=2).fit(scaled)
        projected = []
        for entry in models:
            points = pca.transform(scaler.transform(entry.samples))
            projected.append(ProjectedModel(entry, points, points.mean(axis=0)))

        # Commit only after every calculation succeeds.
        self.models, self.projected = models, projected
        self.scaler, self.pca = scaler, pca

    def _build_user_model(self, model: KeyModel) -> UserModel:
        user_model = UserModel.buildModel(model.to_np(), user_id=self._next_user_id)
        self._next_user_id += 1
        return user_model

    def add_model(self, path) -> bool:
        path = Path(path).resolve()
        if any(entry.path == path for entry in self.models):
            return False
        model = KeyModel.from_file(str(path))
        samples = self._clean_samples(model)
        entry = ImportedModel(
            model,
            samples,
            path.name,
            path,
            user_id=self._next_user_id,
            user_model=self._build_user_model(model),
        )
        self._refit(self.models + [entry])
        return True

    def decision_grid(
        self, entry: ImportedModel, extent: tuple[float, float, float, float],
        resolution: int = 100,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Score the entry's UserModel on a grid over the PC plane.

        Grid points are mapped back through the fitted PCA/scaler to original
        feature space; the returned z array is indexed [ix, iy] to match
        pyqtgraph's col-major isocurve convention.
        """
        if entry.user_model is None or self.pca is None or self.scaler is None:
            return None
        x0, x1, y0, y1 = extent
        xs = np.linspace(x0, x1, resolution)
        ys = np.linspace(y0, y1, resolution)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        points = np.column_stack([gx.ravel(), gy.ravel()])
        features = self.scaler.inverse_transform(self.pca.inverse_transform(points))
        user_model = entry.user_model
        z = np.linalg.norm(
            (features - user_model.median) / user_model.average_center_distance,
            axis=1,
        )
        return xs, ys, z.reshape(len(xs), len(ys))

    def decision_extent(
        self,
        entry: ImportedModel,
        base: tuple[float, float, float, float],
        resolution: int = 100,
        max_expansions: int = 4,
    ) -> tuple[float, float, float, float]:
        """Grow the base extent until the accept region no longer touches the grid border.

        A contour clipped at the grid edge would look cut off; doubling the extent
        around its center guarantees the level set closes inside the grid (or
        stops after max_expansions doublings).
        """
        x0, x1, y0, y1 = base
        for _ in range(max_expansions + 1):
            grid = self.decision_grid(entry, (x0, x1, y0, y1), resolution)
            if grid is None:
                return x0, x1, y0, y1
            z = grid[2]
            border = np.concatenate([z[0], z[-1], z[:, 0], z[:, -1]])
            if border.min() > entry.user_model.threshold:
                return x0, x1, y0, y1
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            x0, x1 = cx - (x1 - x0), cx + (x1 - x0)
            y0, y1 = cy - (y1 - y0), cy + (y1 - y0)
        return x0, x1, y0, y1

    def transform(self, model: KeyModel) -> np.ndarray:
        if self.scaler is None or self.pca is None:
            raise ValueError("Import a reference model before projecting samples.")
        if not model.sample_props:
            return np.empty((0, 2))
        return self.pca.transform(self.scaler.transform(model.to_np()))

    def clear(self):
        self.models.clear()
        self.projected.clear()
        self.scaler = None
        self.pca = None
        self._next_user_id = 1
