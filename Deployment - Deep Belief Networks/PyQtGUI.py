import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel, QSizePolicy, QTextEdit, QSpacerItem, QGridLayout
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui  # Import QtGui
import numpy as np
from datetime import datetime

app = QApplication(sys.argv)

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

in_board = False

try:
    from HALL import HALL_EFFECT
    from BMP180 import BMP180
    from DHT11 import DHT11
    HALL = HALL_EFFECT()
    BMP = BMP180()
    DHT = DHT11()
    in_board = True
    print("Succesfully imported Necessary Packages in RPI")

except Exception as E:
    print("Error: ", E)

window = QMainWindow()
window.setWindowTitle("Hello, PyQt! + 4 Dynamic Fixed-Size PyQtGraph Plots + Console")
window.setGeometry(100, 100, 800, 400)  # Increased width for console

mainWidget = QWidget(window)
mainWidget.setStyleSheet("background-color: white;")  # Set main widget background to white
window.setCentralWidget(mainWidget)

mainLayout = QGridLayout(mainWidget)

# 1. Create a GraphicsLayoutWidget for the plots
graphicsView = pg.GraphicsLayoutWidget()
graphicsView.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
# graphicsView.setStyleSheet("background-color: white;")  # Not reliable

mainLayout.addWidget(graphicsView, 0, 0, 2, 2)

# 2. Add plots to the GraphicsLayoutWidget in a 2x2 grid
temp_plot = graphicsView.addPlot(row=0, col=0, title="Temperature")
humid_plot = graphicsView.addPlot(row=0, col=1, title="Humidity")
graphicsView.nextRow()  # Move to the next row
pressure_plot = graphicsView.addPlot(row=1, col=0, title="Pressure")
wind_plot = graphicsView.addPlot(row=1, col=1, title="Wind Speed")


# Set size policy for each plot
for plot in [temp_plot, humid_plot, pressure_plot, wind_plot]:
    plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

# Set minimum size for each plot (50% of window width and height)
# min_width = (window.width() * 2 / 3) / 2  # Adjusted for console
# min_height = (window.height()) / 2
# for plot in [plot1, plot2, plot3, plot4]:
#     plot.setMinimumSize(int(min_width), int(min_height))  # Convert to int

# Set fixed Y-axis limits for each plot
for plot in [temp_plot, humid_plot, pressure_plot, wind_plot]:
    plot.setYRange(-1.5, 1.5)  # Adjust these values as needed
    plot.getAxis('left').setPen('k')  # Black pen for left axis
    plot.getAxis('bottom').setPen('k') # Black pen for bottom axis
    plot.showAxis('left', True) # Make sure the axis is visible
    plot.showAxis('bottom', True) # Make sure the axis is visible
    plot.getViewBox().setBackgroundColor('w')  # Set ViewBox background to white

# Initialize data for each plot
x1 = np.linspace(0, 10, 100)
temp_data = np.sin(x1) if not in_board else [DHT.readTemperature()] 
curve1 = temp_plot.plot(x1, temp_data, pen='b')

x2 = np.linspace(0, 10, 100)
humid_data = np.cos(x2) if not in_board else [DHT.readHumidity()]
curve2 = humid_plot.plot(x2, humid_data, pen='b')

x3 = np.linspace(0, 10, 100)
pressure_data = np.tan(x3) if not in_board else [BMP.readPressure() * -1 / 1000]
curve3 = pressure_plot.plot(x3, pressure_data, pen='b')

x4 = np.linspace(0, 10, 100)
wind_data = np.exp(-x4) if not in_board else [HALL.readSpeed()]
curve4 = wind_plot.plot(x4, wind_data, pen='b')

# 3. Create a QTextEdit for the console
console = QTextEdit()
console.setReadOnly(True)  # Make it read-only
console.setStyleSheet("background-color: black; color: white;")  # Black background, white text
console.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff) # Remove horizontal scrollbar
console.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff) # Remove vertical scrollbar
spacer = QSpacerItem(1, 1, QSizePolicy.Expanding, QSizePolicy.Expanding)
mainLayout.addItem(spacer, 0, 2)
mainLayout.addWidget(console, 1, 2)  # Console takes up 1/3 of the width

# Initialize last_update_time
last_update_time = datetime.now()

# Update function
def update_plots():
    global x1, temp_data, x2, humid_data, x3, pressure_data, x4, wind_data, last_update_time

    # Measure time difference
    now = datetime.now()
    time_diff = now - last_update_time
    elapsed_ms = time_diff.total_seconds() * 1000  # Convert to milliseconds

    # Update data for each plot (replace with your actual data sources)
    x1 = x1 + 1
    if in_board:
        temp_data.append(DHT.readTemperature())
    else:
        temp_data = np.sin(x1)
    curve1.setData(x1, temp_data)
    current_temp_data = temp_data[-1]  # Get the last y-value

    x2 = x2 + 1  # Different update rate for each plot
    if in_board:
        humid_data.append(DHT.readHumidity())
    else:
        humid_data = np.cos(x2)
    curve2.setData(x2, humid_data)
    current_humid_data = humid_data[-1]

    x3 = x3 + 1
    if in_board:
        pressure_data.append(BMP.readPressure() * -1 / 1000)
    else:
        pressure_data = np.tan(x3)
    curve3.setData(x3, pressure_data)
    current_pressure_data = pressure_data[-1]

    x4 = x4 + 1
    if in_board:
        wind_data.append(HALL.readSpeed())
    else:
        wind_data = np.exp(-x4)
    curve4.setData(x4, wind_data)
    current_wind_data = wind_data[-1]

    # Update last_update_time
    last_update_time = now

    # Create the message
    message = f"[{elapsed_ms:.2f} ms] - Plot 1: {current_temp_data:.2f}, Plot 2: {current_humid_data:.2f}, Plot 3: {current_pressure_data:.2f}, Plot 4: {current_wind_data:.2f}"

    # Append the message to the console
    console.setText(message)

# Timer
timer = QtCore.QTimer()
timer.timeout.connect(update_plots)
timer.start(20)  # Update every 20 milliseconds

window.show()
sys.exit(app.exec_())