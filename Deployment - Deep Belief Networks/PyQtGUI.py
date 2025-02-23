import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QSizePolicy, QTextEdit,QSpacerItem
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import numpy as np
from datetime import datetime

app = QApplication(sys.argv)

window = QMainWindow()
window.setWindowTitle("Hello, PyQt! + 4 Dynamic Fixed-Size PyQtGraph Plots + Console")
window.setGeometry(100, 100, 800, 400)  # Increased width for console

mainWidget = QWidget(window)
window.setCentralWidget(mainWidget)

mainLayout = QtWidgets.QHBoxLayout(mainWidget)  # Horizontal layout for plots and console

# 1. Create a GraphicsLayoutWidget for the plots
graphicsView = pg.GraphicsLayoutWidget()
graphicsView.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
mainLayout.addWidget(graphicsView, 2)  # Plots take up 2/3 of the width

# 2. Add plots to the GraphicsLayoutWidget in a 2x2 grid
plot1 = graphicsView.addPlot(row=0, col=0, title="Plot 1")
plot2 = graphicsView.addPlot(row=0, col=1, title="Plot 2")
graphicsView.nextRow()  # Move to the next row
plot3 = graphicsView.addPlot(row=1, col=0, title="Plot 3")
plot4 = graphicsView.addPlot(row=1, col=1, title="Plot 4")

# Set size policy for each plot
for plot in [plot1, plot2, plot3, plot4]:
    plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

# Set minimum size for each plot (50% of window width and height)
min_width = (window.width() * 2 / 3) / 2  # Adjusted for console
min_height = (window.height()) / 2
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

# 3. Create a QTextEdit for the console
console = QTextEdit()
console.setReadOnly(True)  # Make it read-only
console.setStyleSheet("background-color: black; color: white;")  # Black background, white text
console.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff) # Remove horizontal scrollbar
console.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff) # Remove vertical scrollbar

# 4. Create a layout for the console
consoleLayout = QtWidgets.QVBoxLayout()
consoleLayout.addItem(QtWidgets.QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Expanding)) # Add a spacer item
consoleLayout.addWidget(console)

# 5. Create a widget to hold the console layout
consoleWidget = QWidget()
consoleWidget.setLayout(consoleLayout)
consoleWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
mainLayout.addWidget(consoleWidget, 1)  # Console takes up 1/3 of the width

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

    # Create the message
    message = f"[{elapsed_ms:.2f} ms] - Plot 1: {current_y1:.2f}, Plot 2: {current_y2:.2f}, Plot 3: {current_y3:.2f}, Plot 4: {current_y4:.2f}"

    # Append the message to the console
    console.setText(message)

# Timer
timer = QtCore.QTimer()
timer.timeout.connect(update_plots)
timer.start(20)  # Update every 20 milliseconds

window.show()
sys.exit(app.exec_())