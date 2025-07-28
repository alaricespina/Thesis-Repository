import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QSlider
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import numpy as np

class ScrollableGraph(QMainWindow):
    def __init__(self, data):
        super().__init__()

        self.data = data
        self.data_len = len(data)
        self.visible_range = 50  # Number of data points visible at a time

        self.setWindowTitle("Scrollable Graph")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.plot_widget = pg.PlotWidget()
        self.plot_item = self.plot_widget.plot(self.data)  # Initial plot
        self.layout.addWidget(self.plot_widget)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.data_len - self.visible_range)  # Adjust max
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.update_plot_range)
        self.layout.addWidget(self.slider)

        self.update_plot_range(0)  # Initial plot range

    def update_plot_range(self, value):
        start = value
        end = start + self.visible_range
        self.plot_widget.setXRange(start, end, padding=0)  # padding=0 is important

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Generate some sample data
    data = np.random.normal(size=1000)

    scrollable_graph = ScrollableGraph(data)
    scrollable_graph.show()

    sys.exit(app.exec_())