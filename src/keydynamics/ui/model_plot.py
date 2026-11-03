"""Plot styling and graphics for reference models and live typing."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QColor

from keydynamics.ui.projection import ProjectedModel


@dataclass
class ModelGraphics:
    path: Path
    color: str
    centroid: pg.PlotDataItem
    ellipse: pg.PlotDataItem
    points: pg.PlotDataItem


class ModelPlot(pg.PlotWidget):
    def __init__(self):
        super().__init__(background="#f7f8fa")
        self.references: list[ModelGraphics] = []
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

    def add_reference(self, entry: ProjectedModel):
        palette = ("#3478b8", "#27966f", "#9467bd", "#c18a16", "#298caa", "#9b6b43")
        index = len(self.references)
        if index < len(palette):
            color = palette[index]
        else:
            # Continue with distinct hues outside the red region used for live input.
            hue = (45 + ((index * 0.61803398875) % 1) * 240) / 360
            color = QColor.fromHsvF(hue, 0.65, 0.65).name()
        label = entry.source.path.name
        if any(ref.path.name == entry.source.path.name for ref in self.references):
            label = str(entry.source.path)
        centroid = self.plot(
            pen=None,
            symbol="o",
            symbolSize=12,
            symbolBrush=color,
            symbolPen=None,
            name=label,
        )
        ellipse = self.plot(pen=pg.mkPen(color, width=2))
        ellipse.setZValue(-1)
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
        ellipse.setData(entry.boundary[:, 0], entry.boundary[:, 1])
        points.setData(entry.points[:, 0], entry.points[:, 1])
        self.references.append(
            ModelGraphics(entry.source.path, color, centroid, ellipse, points)
        )

    def show_models(self, models: list[ProjectedModel]):
        self.clear_references()
        for entry in models:
            self.add_reference(entry)

    def clear_references(self):
        for entry in self.references:
            for item in (entry.centroid, entry.ellipse, entry.points):
                self.removeItem(item)
        self.references.clear()

    def show_live(self, points: np.ndarray):
        self.live_curve.setData(points[:, 0], points[:, 1])
