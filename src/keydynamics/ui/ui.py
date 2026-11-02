import sys
import time
from pathlib import Path
from queue import Queue
from uuid import uuid4

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from keydynamics.datagenerator import DataGenerator
from keydynamics.models.model import KeyModel


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Keystroke model — 2D PCA")
        self.resize(800, 650)
        self.model = None
        self.scaler = None
        self.pca = None

        self.import_button = QPushButton("Import .npy model…")
        self.import_button.clicked.connect(self.import_model)
        self.status = QLabel("Import a model to display its samples.")
        self.status.setWordWrap(True)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True)
        self.plot.setLabel("bottom", "Principal component 1")
        self.plot.setLabel("left", "Principal component 2")
        self.curve = self.plot.plot(pen=None, symbol="o", symbolBrush="c")

        self.textbox = QPlainTextEdit()
        font = self.textbox.font()
        font.setPointSize(20)
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

        self.events = Queue()
        self.generator = DataGenerator(
            participant_id=0, session_id=uuid4(), memory=self.events
        )

        self.textbox.installEventFilter(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)
        layout.addWidget(self.import_button)
        layout.addWidget(self.status)
        layout.addWidget(self.plot, stretch=1)
        layout.addWidget(self.textbox)

    def eventFilter(self, watched, event):
        if (
            watched is self.textbox
            and event.type()
            in (
                QEvent.Type.KeyPress,
                QEvent.Type.KeyRelease,
            )
            and not event.isAutoRepeat()
        ):
            key_code = event.nativeScanCode() or event.key()
            timestamp = time.perf_counter()
            text = event.text() or None

            if event.type() == QEvent.Type.KeyPress:
                self.generator.registerPress(key_code, timestamp, text)
            else:
                self.generator.registerRelease(key_code, timestamp, text)

        # Let the text box process the event normally.
        return super().eventFilter(watched, event)

    def import_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import keystroke model", "", "NumPy models (*.npy)"
        )
        if not path:
            return

        try:
            self.load_model(path)
        except (OSError, ValueError, TypeError, EOFError, IndexError) as error:
            QMessageBox.warning(self, "Cannot import model", str(error))

    def load_model(self, path):
        model = KeyModel.from_file(path)
        if not model.sample_props:
            raise ValueError("The model contains no samples.")
        if not np.isfinite(model.to_np()).all():
            raise ValueError("Model features must be finite numbers.")

        samples = model.clean().to_numpy(dtype=float)
        if samples.shape[0] < 2:
            raise ValueError(
                "At least two samples must remain after cleaning for 2D PCA."
            )

        scaler = StandardScaler()
        scaled = scaler.fit_transform(samples)
        if not np.any(np.var(scaled, axis=0) > 0):
            raise ValueError("The samples have no variation to visualize.")
        pca = PCA(n_components=2)
        points = pca.fit_transform(scaled)

        self.model = model
        self.scaler = scaler
        self.pca = pca
        self.curve.setData(points[:, 0], points[:, 1])
        self.plot.autoRange()
        self.status.setText(
            f"{Path(path).name} — {len(samples)} / {len(model.sample_props)} samples "
            f"after cleaning · explained variance: "
            f"PC1 {pca.explained_variance_ratio_[0]:.1%}, "
            f"PC2 {pca.explained_variance_ratio_[1]:.1%}"
        )


def main():
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
