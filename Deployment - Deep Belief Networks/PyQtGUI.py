import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QSizePolicy
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import numpy as np
from datetime import datetime

app = QApplication(sys.argv)

window = QMainWindow()
window.setWindowTitle("Hello, PyQt! + 4 Dynamic Fixed-Size PyQtGraph Plots")
window.setGeometry(100, 100, 800, 600)  # Increased height

mainWidget = QWidget(window)
window.setCentralWidget(mainWidget)

mainLayout = QtWidgets.QVBoxLayout(mainWidget)

label = QLabel("Implementation of a Deep Belief Network with Sensor Correction Algorithm to predict Weather on a Raspberry Pi")
mainLayout.addWidget(label)

# Create a GraphicsLayoutWidget
graphicsView = pg.GraphicsLayoutWidget()
mainLayout.addWidget(graphicsView)

# Add plots to the GraphicsLayoutWidget in a 2x2 grid
plot1 = graphicsView.addPlot(row=0, col=0, title="Plot 1")
plot2 = graphicsView.addPlot(row=0, col=1, title="Plot 2")
graphicsView.nextRow()  # Move to the next row
plot3 = graphicsView.addPlot(row=1, col=0, title="Plot 3")
plot4 = graphicsView.addPlot(row=1, col=1, title="Plot 4")

# Set size policy for each plot
for plot in [plot1, plot2, plot3, plot4]:
    plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

# Set minimum size for each plot (50% of window width and height)
min_width = window.width() / 2
min_height = (window.height() - label.height()) / 2  # Account for label height
for plot in [plot1, plot2, plot3, plot4]:
    plot.setMinimumSize(min_width, min_height)

# Set fixed Y-axis limits for each plot
for plot in [plot1, plot2, plot3, plot4]:
    plot.setYRange(-1.5, 1.5)  # Adjust these values as needed

# Initialize data for each plot
x1 = np.linspace(0, 10, 100)
y1 = np.sin(x1)
curve1 = plot1.plot(x1, y1, pen='r')

x2 = np.linspace(0, 10, 100)
y2 = np.cos(x2)
curve2 = plot2.plot(x2, y2, pen='g')

x3 = np.linspace(0, 10, 100)
y3 = np.tan(x3)
curve3 = plot3.plot(x3, y3, pen='b')

x4 = np.linspace(0, 10, 100)
y4 = np.exp(-x4)
curve4 = plot4.plot(x4, y4, pen='y')

# Initialize last_update_time
last_update_time = datetime.now()

# Update function
def update_plots():
    global x1, y1, x2, y2, x3, y3, x4, y4, last_update_time

    # Measure time difference
    now = datetime.now()
    time_diff = now - last_update_time
    elapsed_ms = time_diff.total_seconds() * 1000  # Convert to milliseconds

    # Update data for each plot (replace with your actual data sources)
    x1 = x1 + 0.1
    y1 = np.sin(x1)
    curve1.setData(x1, y1)
    current_y1 = y1[-1]  # Get the last y-value

    x2 = x2 + 0.15  # Different update rate for each plot
    y2 = np.cos(x2)
    curve2.setData(x2, y2)
    current_y2 = y2[-1]

    x3 = x3 + 0.2
    y3 = np.tan(x3)
    curve3.setData(x3, y3)
    current_y3 = y3[-1]

    x4 = x4 + 0.25
    y4 = np.exp(-x4)
    curve4.setData(x4, y4)
    current_y4 = y4[-1]

    # Update last_update_time
    last_update_time = now

    # Print the elapsed time and current y-values
    print(f"Update interval: {elapsed_ms:.2f} ms | Plot 1: {current_y1:.2f}, Plot 2: {current_y2:.2f}, Plot 3: {current_y3:.2f}, Plot 4: {current_y4:.2f}")

# Timer
timer = QtCore.QTimer()
timer.timeout.connect(update_plots)
timer.start(10)  # Update every 100 milliseconds

window.show()
sys.exit(app.exec_())