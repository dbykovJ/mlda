"""Plot styling and graphics for reference models and live typing."""

from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QTransform

from keydynamics.ui.projection import ProjectedModel


@dataclass
class ModelGraphics:
    name: str
    color: str
    centroid: pg.PlotDataItem
    points: pg.PlotDataItem
    contour: object | None = None


class ModelPlot(pg.PlotWidget):
    def __init__(self):
        super().__init__(background="#f7f8fa")
        self.references: list[ModelGraphics] = []
        self.projection = None
        self.contour_resolution = 100
        self.global_mode = False
        self.global_k = 5
        self.global_image = None
        self.setAspectLocked(True, ratio=1)
        for position in ("bottom", "left"):
            axis = self.getAxis(position)
            axis.setPen("#667085")
            axis.setTextPen("#344054")
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel("bottom", "Principal component 1", color="#344054")
        self.setLabel("left", "Principal component 2", color="#344054")
        self.addLegend(brush="#ffffff", pen="#d0d5dd", labelTextColor="#344054")
        self.live_curve = self.plot(
            pen=None,
            symbol="o",
            symbolSize=10,
            symbolBrush="red",
            symbolPen=None,
            name="Live typing",
        )
        self.saved_live: dict[tuple[str, bool], pg.PlotDataItem] = {}
        self.colors: dict[str, str] = {}

    def _color_for(self, name: str) -> str:
        # One registry for model and live-label names so colors never collide,
        # regardless of import/save order or re-draws.
        if name not in self.colors:
            palette = ("#3478b8", "#27966f", "#9467bd", "#c18a16", "#298caa", "#9b6b43")
            index = len(self.colors)
            if index < len(palette):
                self.colors[name] = palette[index]
            else:
                # Continue with distinct hues outside the red region used for live input.
                hue = (45 + ((index * 0.61803398875) % 1) * 240) / 360
                self.colors[name] = QColor.fromHsvF(hue, 0.65, 0.65).name()
        return self.colors[name]

    def add_reference(self, entry: ProjectedModel):
        color = self._color_for(entry.source.name)
        label = entry.source.name
        if any(ref.name == entry.source.name for ref in self.references):
            label = f"{entry.source.name} ({len(self.references) + 1})"
        centroid = self.plot(
            pen=None,
            symbol="o",
            symbolSize=12,
            symbolBrush=color,
            symbolPen=None,
            name=label,
        )
        faint = pg.mkColor(color)
        faint.setAlpha(65)
        points = self.plot(
            pen=None,
            symbol="o",
            symbolSize=4,
            symbolBrush=faint,
            symbolPen=None,
        )
        points.setZValue(-2)
        centroid.setData([entry.center[0]], [entry.center[1]])
        points.setData(entry.points[:, 0], entry.points[:, 1])
        contour = self._decision_contour(entry, color)
        self.references.append(
            ModelGraphics(entry.source.name, color, centroid, points, contour)
        )

    def _decision_contour(self, entry: ProjectedModel, color):
        if self.global_mode:
            return None
        if self.projection is None or entry.source.user_model is None:
            return None
        # Grid around this model's own points so a large model cannot
        # stretch the extent and starve the contour of grid resolution.
        low = entry.points.min(axis=0)
        high = entry.points.max(axis=0)
        pad = np.maximum((high - low) * 0.1, 0.05)
        base = (low[0] - pad[0], high[0] + pad[0],
                low[1] - pad[1], high[1] + pad[1])
        extent = self.projection.decision_extent(
            entry.source, base, self.contour_resolution
        )
        grid = self.projection.decision_grid(
            entry.source, extent, self.contour_resolution
        )
        if grid is None:
            return None
        xs, ys, z = grid
        contour = pg.IsocurveItem(
            data=z,
            level=entry.source.user_model.threshold,
            pen=pg.mkPen(color, width=2),
            axisOrder="col-major",
        )
        contour.setPos(xs[0], ys[0])
        contour.setTransform(QTransform.fromScale(xs[1] - xs[0], ys[1] - ys[0]))
        contour.setZValue(-1)
        self.addItem(contour)
        return contour

    def _global_regions(self, extent):
        grid = self.projection.global_grid(
            extent,
            self.contour_resolution,
            k=self.global_k,
        )
        if grid is None:
            return None
        xs, ys, field = grid
        image = np.zeros((*field.shape, 4), dtype=np.uint8)
        for user_id in np.unique(field):
            name = next(
                (e.name for e in self.projection.models if e.user_id == user_id),
                None,
            )
            color = QColor(self._color_for(name) if name else "#888888")
            color.setAlpha(60)
            image[field == user_id] = (
                color.red(), color.green(), color.blue(), color.alpha(),
            )
        item = pg.ImageItem(image=image)
        item.setRect(QRectF(xs[0], ys[0], xs[-1] - xs[0], ys[-1] - ys[0]))
        item.setZValue(-3)
        self.addItem(item)
        return item

    def show_models(self, models: list[ProjectedModel]):
        self.clear_references()
        if self.global_mode and models:
            all_points = np.vstack([entry.points for entry in models])
            low = all_points.min(axis=0)
            high = all_points.max(axis=0)
            pad = np.maximum((high - low) * 0.1, 0.05)
            extent = (low[0] - pad[0], high[0] + pad[0],
                      low[1] - pad[1], high[1] + pad[1])
            self.global_image = self._global_regions(extent)
        for entry in models:
            self.add_reference(entry)

    def clear_references(self):
        if self.global_image is not None:
            self.removeItem(self.global_image)
            self.global_image = None
        for entry in self.references:
            items = [entry.centroid, entry.points]
            if entry.contour is not None:
                items.append(entry.contour)
            for item in items:
                self.removeItem(item)
        self.references.clear()
        for curve in self.saved_live.values():
            self.removeItem(curve)
        self.saved_live.clear()

    def show_live(
        self,
        points: np.ndarray,
        saved_names: list[str | None] | None = None,
        accepted: list[bool] | None = None,
    ):
        n = len(points)
        if saved_names is None:
            saved_names = [None] * n
        if accepted is None:
            accepted = [False] * n
        unsaved = [i for i, name in enumerate(saved_names) if name is None]
        if unsaved:
            brushes = ["green" if accepted[i] else "red" for i in unsaved]
            self.live_curve.setData(
                points[unsaved, 0], points[unsaved, 1], symbolBrush=brushes
            )
        else:
            self.live_curve.setData([], [])

        names = {name for name in saved_names if name is not None}
        for key in list(self.saved_live):
            if key[0] not in names:
                self.removeItem(self.saved_live.pop(key))
        for name in names:
            color = self._color_for(name)
            for is_accepted, symbol, legend in ((True, "o", name), (False, "x", None)):
                indices = [
                    i
                    for i, sample_name in enumerate(saved_names)
                    if sample_name == name and bool(accepted[i]) == is_accepted
                ]
                key = (name, is_accepted)
                curve = self.saved_live.get(key)
                if curve is None:
                    curve = self.plot(
                        pen=None,
                        symbol=symbol,
                        symbolSize=10,
                        symbolBrush=color,
                        symbolPen=pg.mkPen(color, width=2),
                        name=legend,
                    )
                    self.saved_live[key] = curve
                curve.setData(points[indices, 0], points[indices, 1])

    def clear_labels(self):
        self.colors.clear()
