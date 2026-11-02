import sys

import pyqtgraph as pg
from PySide6.QtWidgets import (
    QApplication,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live plot")
        self.resize(600, 400)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True)
        self.curve = self.plot.plot(pen="c", symbol="o")

        self.textbox = QLineEdit()
        self.textbox.setPlaceholderText("Numbers, e.g. 1, 4, 2, 8")

        layout = QVBoxLayout(self)
        layout.addWidget(self.plot, stretch=1)
        layout.addWidget(self.textbox)

        self.textbox.textChanged.connect(self.update_plot)
        self.textbox.setText("1, 4, 2, 8")

    def update_plot(self, text):
        if not text.strip():
            self.curve.setData([], [])
            return

        try:
            values = [float(value.strip()) for value in text.split(",")]
        except ValueError:
            # Keep the previous plot while input is incomplete.
            return

        self.curve.setData(list(range(len(values))), values)


app = QApplication(sys.argv)
window = Window()
window.show()
sys.exit(app.exec())
