"""Application entry point and window layout."""

import os
import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from keydynamics.ui.model_plot import ModelPlot
from keydynamics.ui.projection import ModelProjection
from keydynamics.ui.typing_recorder import TypingRecorder


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Keystroke model — 2D PCA")
        self.resize(800, 650)
        self.projection = ModelProjection()

        self.import_button = QPushButton("Import .npy models…")
        self.import_button.clicked.connect(self.import_model)
        self.clear_button = QPushButton("Clear models")
        self.clear_button.clicked.connect(self.clear_models)
        self.user_mode = QRadioButton("User mode")
        self.user_mode.setChecked(True)
        self.global_mode = QRadioButton("Global mode")
        self.global_mode.toggled.connect(self.on_mode_changed)
        self.status = QLabel("Import models to compare their distributions.")
        self.status.setWordWrap(True)

        self.plot = ModelPlot()
        self.plot.projection = self.projection
        self.live_status = QLabel(
            "Live typing: 10 completed keystrokes needed per sample."
        )
        self.live_status.setWordWrap(True)

        self.textbox = QPlainTextEdit()
        # Timing features require individual key events, not compressed text input.
        self.textbox.setAttribute(Qt.WidgetAttribute.WA_KeyCompression, False)
        # Measure physical typing rather than text committed by an input method.
        self.textbox.setAttribute(Qt.WidgetAttribute.WA_InputMethodEnabled, False)
        self.textbox.viewport().setAttribute(
            Qt.WidgetAttribute.WA_InputMethodEnabled, False
        )
        font = self.textbox.font()
        font.setPointSize(15)
        self.textbox.setFont(font)
        self.textbox.setStyleSheet("""
            QPlainTextEdit {
                border: 2px solid #888;
                border-radius: 16px;
                padding: 12px;
            }
            QPlainTextEdit:focus {
                border-color: #3399ff;
            }
        """)
        self.textbox.setFixedHeight(self.textbox.fontMetrics().lineSpacing() * 5 + 40)
        self.textbox.setPlaceholderText("Type here…")

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Model name…")
        self.save_button = QPushButton("Save as model")
        self.save_button.clicked.connect(self.save_live_model)
        self.clear_input_button = QPushButton("Clear input")
        self.clear_input_button.clicked.connect(self.clear_input)

        self.recorder = TypingRecorder(self.textbox)
        self.recorder.model_changed.connect(self.update_live_sample)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        self.content = QWidget()
        self.content.setMaximumWidth(800)
        outer.addWidget(
            self.content, stretch=1, alignment=Qt.AlignmentFlag.AlignHCenter
        )
        layout = QVBoxLayout(self.content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        buttons = QHBoxLayout()
        buttons.addWidget(self.import_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch(1)
        buttons.addWidget(self.user_mode)
        buttons.addWidget(self.global_mode)
        layout.addLayout(buttons)
        layout.addWidget(self.status)
        layout.addWidget(self.plot, stretch=1)
        layout.addWidget(self.live_status)
        layout.addWidget(self.textbox)
        save_row = QHBoxLayout()
        save_row.addWidget(self.name_edit, stretch=1)
        save_row.addWidget(self.save_button)
        save_row.addWidget(self.clear_input_button)
        layout.addLayout(save_row)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        margins = self.layout().contentsMargins()
        width = max(0, min(800, self.width() - margins.left() - margins.right()))
        self.content.setFixedWidth(width)

    def update_live_sample(self, model, completed):
        if not model.sample_props:
            self.live_status.setText(
                f"Live typing: {completed}/10 completed keystrokes in this sample."
            )
            return
        if self.projection.scaler is None or self.projection.pca is None:
            self.live_status.setText(
                "Live samples recorded. Import a model to project them."
            )
            return

        points = self.projection.transform(model)
        self.plot.show_live(
            points, self.recorder.sample_names(), self._accepted_flags(model)
        )
        if len(points) and self._outside_view(points):
            self.plot.autoRange()
        self.live_status.setText(
            f"Live typing: {len(points)} samples · {completed} completed keystrokes "
            f"in this sample · pause over 3 seconds to start another."
        )

    def on_mode_changed(self):
        self.plot.global_mode = self.global_mode.isChecked()
        self.plot.show_models(self.projection.projected)
        self.update_live_sample(self.recorder.live_model, self.recorder.completed)

    def _outside_view(self, points: np.ndarray) -> bool:
        (xmin, xmax), (ymin, ymax) = self.plot.getViewBox().viewRange()
        return bool(
            np.any(points[:, 0] < xmin)
            or np.any(points[:, 0] > xmax)
            or np.any(points[:, 1] < ymin)
            or np.any(points[:, 1] > ymax)
        )

    def _accepted_flags(self, model) -> list[bool]:
        # A live sample is accepted if any imported user model's decision
        # boundary contains it; computed in original feature space.
        user_models = [
            entry.user_model
            for entry in self.projection.models
            if entry.user_model is not None
        ]
        features = model.to_np()
        flags = [False] * len(model.sample_props)
        for user_model in user_models:
            scores = np.linalg.norm(
                (features - user_model.median) / user_model.average_center_distance,
                axis=1,
            )
            flags = [
                flag or bool(score <= user_model.threshold)
                for flag, score in zip(flags, scores)
            ]
        return flags

    def import_model(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import keystroke models", "", "NumPy models (*.npy)"
        )
        errors = []
        for path in paths:
            try:
                self.load_model(path)
            except (OSError, ValueError, TypeError, EOFError, IndexError) as error:
                errors.append(f"{Path(path).name}: {error}")
        if errors:
            QMessageBox.warning(self, "Cannot import models", "\n".join(errors))

    def load_model(self, path):
        if not self.projection.add_model(path):
            return
        self.plot.show_models(self.projection.projected)
        self.update_live_sample(self.recorder.live_model, self.recorder.completed)
        self.plot.autoRange()
        self.update_status()

    def save_live_model(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Cannot save model", "Enter a name for the model.")
            return
        if (
            any(entry.name == name for entry in self.projection.models)
            or name in self.plot.colors
        ):
            QMessageBox.warning(
                self, "Cannot save model", f'A model named "{name}" already exists.'
            )
            return
        # Live samples stay plain labeled points; they are not added to the
        # projection, so no PCA refit, centroid, or decision contour.
        self.recorder.mark_saved(name)
        self.update_live_sample(self.recorder.live_model, self.recorder.completed)
        self.name_edit.clear()
        self.textbox.clear()

    def clear_input(self):
        self.recorder.clear_unsaved()
        self.textbox.clear()

    def update_status(self):
        counts = "; ".join(
            f"{entry.name}: {len(entry.samples)}/{len(entry.model.sample_props)} samples"
            for entry in self.projection.models
        )
        pca = self.projection.pca
        self.status.setText(
            f"{len(self.projection.models)} models · decision contours · {counts}\n"
            f"Shared PCA explained variance: PC1 {pca.explained_variance_ratio_[0]:.1%}, "
            f"PC2 {pca.explained_variance_ratio_[1]:.1%}"
        )

    def clear_models(self):
        self.projection.clear()
        self.recorder.reset()
        self.plot.clear_references()
        self.plot.clear_labels()
        self.plot.show_live(np.empty((0, 2)))
        self.status.setText("Import models to compare their distributions.")
        self.update_live_sample(self.recorder.live_model, self.recorder.completed)


def main():
    if (
        sys.platform == "linux"
        and (
            os.environ.get("WAYLAND_DISPLAY")
            or os.environ.get("XDG_SESSION_TYPE") == "wayland"
        )
        and "QT_IM_MODULE" not in os.environ
        and "QT_IM_MODULES" not in os.environ
    ):
        # Wayland text-input can consume key events needed for timing measurements.
        os.environ["QT_IM_MODULE"] = "compose"

    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
