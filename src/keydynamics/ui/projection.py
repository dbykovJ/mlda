"""Model loading and shared PCA calculations, independent of Qt."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from keydynamics.models.model import KeyModel


def calculate_model_distribution(
    model: KeyModel,
    scaler: StandardScaler,
    pca: PCA,
    confidence: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the cleaned model's centroid and ellipse boundary in shared PCA space.

    The ellipse describes sample spread assuming a 2D Gaussian distribution;
    it is not a confidence region for the mean. Scaler and PCA must be fitted.
    """
    if not 0 < confidence < 1:
        raise ValueError("Confidence must be between zero and one.")
    if len(model.sample_props) < 2:
        raise ValueError("At least two samples are needed for a distribution.")
    samples = model.clean().to_numpy(dtype=float)
    if len(samples) < 2 or not np.isfinite(samples).all():
        raise ValueError("At least two finite samples must remain after cleaning.")
    points = pca.transform(scaler.transform(samples))
    if points.shape[1] != 2:
        raise ValueError("The PCA must have two components.")
    center = points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points, rowvar=False))
    radii = np.sqrt(np.maximum(eigenvalues, 0) * chi2.ppf(confidence, df=2))
    angles = np.linspace(0, 2 * np.pi, 181)
    circle = np.stack([np.cos(angles), np.sin(angles)])
    boundary = center + (eigenvectors @ (radii[:, None] * circle)).T
    return center, boundary


@dataclass
class ImportedModel:
    path: Path
    model: KeyModel
    samples: np.ndarray


@dataclass
class ProjectedModel:
    source: ImportedModel
    points: np.ndarray
    center: np.ndarray
    boundary: np.ndarray


class ModelProjection:
    def __init__(self):
        self.models: list[ImportedModel] = []
        self.projected: list[ProjectedModel] = []
        self.scaler = None
        self.pca = None

    def add_model(self, path) -> bool:
        path = Path(path).resolve()
        if any(entry.path == path for entry in self.models):
            return False
        model = KeyModel.from_file(str(path))
        if not model.sample_props:
            raise ValueError("The model contains no samples.")
        if not np.isfinite(model.to_np()).all():
            raise ValueError("Model features must be finite numbers.")
        samples = model.clean().to_numpy(dtype=float)
        if len(samples) < 2 or not np.isfinite(samples).all():
            raise ValueError("At least two finite samples must remain after cleaning.")

        models = self.models + [ImportedModel(path, model, samples)]
        combined = np.vstack([entry.samples for entry in models])
        scaler = StandardScaler()
        scaled = scaler.fit_transform(combined)
        if not np.any(np.var(scaled, axis=0) > 0):
            raise ValueError("The samples have no variation to visualize.")
        pca = PCA(n_components=2).fit(scaled)
        projected = []
        for entry in models:
            center, boundary = calculate_model_distribution(entry.model, scaler, pca)
            points = pca.transform(scaler.transform(entry.samples))
            projected.append(ProjectedModel(entry, points, center, boundary))

        # Commit only after every calculation succeeds.
        self.models, self.projected = models, projected
        self.scaler, self.pca = scaler, pca
        return True

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
